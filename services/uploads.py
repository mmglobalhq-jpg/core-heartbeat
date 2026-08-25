"""Filesystem I/O for the Settings -> File Upload feature.

Writes user-uploaded files to a directory on the mini PC's **Windows** drive
(``C:\\Users\\MMGlobal\\Uploads``, bind-mounted into this container at
``/data/uploads``) so they are reachable from Explorer, not just from WSL.

Everything here is deliberately paranoid, because this is the platform's first
endpoint that writes attacker-influenced names and bytes to local disk, and it is
reachable from ``chat.mmglobal.us`` -- an edge with **no Cloudflare Access** in
front of it (see 10-network-and-trust-boundaries.md). The gateway's admin gate is
the only thing between the open internet and this module.

Four behaviours here exist because the target is a 9p/DrvFs mount, verified on the
box 2026-08-25 rather than assumed:

* **``chmod`` does nothing.** ``/mnt/c`` is mounted ``uid=1000,gid=1000`` with no
  ``metadata`` option, so every file lands ``0777`` whatever we ask for. Setting a
  mode here would be a comforting no-op; we do not pretend. Fixing it properly
  means ``options="metadata"`` in ``/etc/wsl.conf`` plus a WSL restart, which stops
  every container on the box.
* **Windows-illegal names must be rewritten** (:func:`safe_name`). 9p happily
  creates ``CON.txt``; Windows then *lists* it in Explorer but cannot open or
  delete it (``Get-Content`` -> ``ItemNotFoundException``). A file that uploads
  "successfully" and is permanently unopenable is the worst outcome available for
  a feature whose entire point is Windows access.
* **DrvFs is case-insensitive**, so collision detection is NTFS-correct for free:
  ``Path("A.TXT").exists()`` is True when ``a.txt`` is present. We never overwrite.
* **The mount can go missing.** Docker *creates* a bind-mount source path that
  isn't there, so a WSL boot without ``/mnt/c`` would silently redirect every
  upload into an ext4 directory invisible from Windows. :func:`ensure_root` fails
  the request loudly on a missing sentinel instead.

All disk calls are blocking (:mod:`os` / :mod:`pathlib`); the async entry point
:func:`store_stream` offloads each write with ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

# --- configuration (env-overridable, read at call time) ---------------------

UPLOAD_ROOT_ENV = "UPLOAD_ROOT"
DEFAULT_UPLOAD_ROOT = "/data/uploads"

MAX_UPLOAD_BYTES_ENV = "MAX_UPLOAD_BYTES"
# 100 MB: Cloudflare's free plan rejects a larger request body at the edge, so a
# higher cap here would only turn a clear in-app error into an opaque edge 413.
DEFAULT_MAX_UPLOAD_BYTES = 100 * 1024 * 1024

UPLOAD_TOTAL_BUDGET_BYTES_ENV = "UPLOAD_TOTAL_BUDGET_BYTES"
DEFAULT_UPLOAD_TOTAL_BUDGET_BYTES = 50 * 1024 * 1024 * 1024  # 50 GB

UPLOAD_FREE_SPACE_FLOOR_ENV = "UPLOAD_FREE_SPACE_FLOOR_BYTES"
DEFAULT_FREE_SPACE_FLOOR = 5 * 1024 * 1024 * 1024  # keep 5 GB of C: free

# Written into the directory by hand at install time; its absence means we are NOT
# looking at the Windows folder (see module docstring).
SENTINEL = ".uploads-root"

# Partial uploads land here first and are renamed into place only once complete,
# so a reader never sees a half-written file under its real name.
PART_SUFFIX = ".part"

MAX_NAME_BYTES = 200

# Reserved DOS device names. Windows applies these per *stem*, case-insensitively,
# with or without an extension -- "con.txt" is as unusable as "CON".
_RESERVED_STEMS = {
    "con", "prn", "aux", "nul", "conin$", "conout$",
    *(f"com{i}" for i in range(10)),
    *(f"lpt{i}" for i in range(10)),
}

# Characters NTFS refuses, plus C0 controls. "/" is handled by basename() first.
_ILLEGAL_CHARS = re.compile(r'[<>:"\\|?*\x00-\x1f]')


class UploadError(Exception):
    """A refusal with the HTTP status the gateway should return."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def upload_root() -> Path:
    return Path(os.environ.get(UPLOAD_ROOT_ENV) or DEFAULT_UPLOAD_ROOT)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def max_upload_bytes() -> int:
    return _int_env(MAX_UPLOAD_BYTES_ENV, DEFAULT_MAX_UPLOAD_BYTES)


def total_budget_bytes() -> int:
    return _int_env(UPLOAD_TOTAL_BUDGET_BYTES_ENV, DEFAULT_UPLOAD_TOTAL_BUDGET_BYTES)


def free_space_floor() -> int:
    return _int_env(UPLOAD_FREE_SPACE_FLOOR_ENV, DEFAULT_FREE_SPACE_FLOOR)


# --- name hardening ---------------------------------------------------------


def safe_name(raw: str) -> tuple[str, bool]:
    """Reduce a client-supplied filename to one that is safe *and* openable on NTFS.

    Returns ``(name, rewritten)``; ``rewritten`` is True when the result differs
    from what the caller asked for, so the UI can say so rather than silently
    storing something else. Never raises, never returns an empty name.
    """
    original = raw

    # Path components first: only ever a basename, on either separator. This alone
    # defeats "../../etc/passwd" and "..\\..\\x"; the containment assert in
    # resolve_target() is the second, independent check.
    name = raw.replace("\\", "/").split("/")[-1]

    name = _ILLEGAL_CHARS.sub("_", name)

    # Leading dots would hide the file from Explorer and could target the sentinel.
    name = name.lstrip(".")

    # NTFS silently drops trailing dots/spaces on create, so a name ending in one
    # resolves to a *different* file than the one we think we wrote.
    name = name.rstrip(". ")

    stem, dot, ext = name.partition(".")
    if stem.lower() in _RESERVED_STEMS:
        stem = f"{stem}_file"
        name = f"{stem}{dot}{ext}"

    # Byte-cap the stem, not the whole name, so the extension survives -- a
    # truncated extension would change how Windows opens the file.
    if len(name.encode("utf-8", "ignore")) > MAX_NAME_BYTES:
        ext_b = f"{dot}{ext}".encode("utf-8", "ignore")[: MAX_NAME_BYTES // 2]
        keep = MAX_NAME_BYTES - len(ext_b)
        stem_b = stem.encode("utf-8", "ignore")[:keep]
        name = stem_b.decode("utf-8", "ignore") + ext_b.decode("utf-8", "ignore")
        name = name.rstrip(". ")

    if not name:
        name = "upload"

    return name, name != original


def ensure_root() -> Path:
    """Return the upload root, or raise 503 if it is not the real Windows folder.

    The sentinel check is the guard against Docker having auto-created the
    bind-mount source on a boot where /mnt/c was not mounted: uploads would
    otherwise land in an ext4 directory nothing on the Windows side can see.
    """
    root = upload_root()
    if not (root / SENTINEL).exists():
        raise UploadError(
            503,
            "upload directory is not mounted (sentinel missing) - refusing to write "
            "where Windows cannot see the file",
        )
    return root


def resolve_target(name: str) -> Path:
    """Resolve ``name`` inside the root and prove the result did not escape it."""
    root = upload_root().resolve()
    target = (root / name).resolve()
    if target == root or not target.is_relative_to(root):
        raise UploadError(400, "invalid filename")
    if target.name == SENTINEL:
        raise UploadError(400, "reserved filename")
    return target


def unique_target(target: Path) -> Path:
    """First free name in the ``name (2).ext`` series. Never overwrites.

    DrvFs is case-insensitive, so ``exists()`` also catches ``Report.pdf`` vs
    ``report.pdf`` -- matching how the file would behave in Explorer.
    """
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    for n in range(2, 1000):
        candidate = target.with_name(f"{stem} ({n}){suffix}")
        if not candidate.exists():
            return candidate
    raise UploadError(409, "too many files with that name")


# --- capacity ---------------------------------------------------------------


def dir_size(root: Path) -> int:
    total = 0
    for entry in root.iterdir():
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def guard_capacity(root: Path) -> None:
    """Refuse before writing when C: is nearly full or the folder is over budget."""
    try:
        free = shutil.disk_usage(root).free
    except OSError:
        free = None
    if free is not None and free < free_space_floor():
        raise UploadError(507, "not enough free disk space on the mini PC")
    if dir_size(root) >= total_budget_bytes():
        raise UploadError(
            507, "the upload folder has reached its size budget - delete some files first"
        )


# --- write / list / delete --------------------------------------------------


async def store_stream(chunks: AsyncIterator[bytes], filename: str) -> dict:
    """Stream a request body to disk under a hardened name. Returns a result dict.

    Bytes are counted as they arrive and the write is abandoned the moment the cap
    is crossed, so an oversized upload costs us the cap and not the whole file.
    """
    root = ensure_root()
    guard_capacity(root)

    name, rewritten = safe_name(filename)
    target = unique_target(resolve_target(name))
    part = target.with_name(target.name + PART_SUFFIX)
    cap = max_upload_bytes()

    written = 0
    try:
        fh = await asyncio.to_thread(open, part, "wb")
        try:
            async for chunk in chunks:
                if not chunk:
                    continue
                written += len(chunk)
                if written > cap:
                    raise UploadError(
                        413, f"file exceeds the {cap // (1024 * 1024)} MB limit"
                    )
                await asyncio.to_thread(fh.write, chunk)
        finally:
            await asyncio.to_thread(fh.close)
    except UploadError:
        await asyncio.to_thread(_unlink_quietly, part)
        raise
    except Exception as exc:
        await asyncio.to_thread(_unlink_quietly, part)
        raise UploadError(500, f"write failed: {type(exc).__name__}") from exc

    if written == 0:
        await asyncio.to_thread(_unlink_quietly, part)
        raise UploadError(400, "empty file")

    # Atomic within the directory (verified on DrvFs): readers see either nothing
    # or the finished file, never a partial one under its real name.
    await asyncio.to_thread(os.replace, part, target)

    # Windows Defender scans writes to C: and can quarantine a file *after* the
    # rename returns successfully. Report that as a failure rather than a success
    # the user cannot act on.
    if not await asyncio.to_thread(target.exists):
        raise UploadError(
            502, "file was stored but immediately removed by Windows Defender"
        )

    # NOTE: no chmod. The DrvFs mount has no `metadata` option, so mode is fixed at
    # 0777 and a chmod call here would be a no-op that reads like a guarantee.
    return {
        "stored_as": target.name,
        "original_name": filename,
        "rewritten": rewritten or target.name != name,
        "size_bytes": written,
        "windows_path": f"C:\\Users\\MMGlobal\\Uploads\\{target.name}",
    }


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def list_uploads() -> list[dict]:
    """Every file in the folder, newest first.

    A plain directory read, so files dropped in from Windows are listed too -- the
    folder is the source of truth, not a database. Hides the sentinel and any
    in-flight ``.part`` file.
    """
    root = ensure_root()
    entries: list[dict] = []
    for entry in root.iterdir():
        if entry.name == SENTINEL or entry.name.endswith(PART_SUFFIX):
            continue
        try:
            if not entry.is_file():
                continue
            st = entry.stat()
        except OSError:
            continue
        entries.append(
            {"name": entry.name, "size_bytes": st.st_size, "modified": int(st.st_mtime)}
        )
    entries.sort(key=lambda e: e["modified"], reverse=True)
    return entries


def delete_upload(name: str) -> None:
    """Delete one file from the folder. Same containment guard as the write path."""
    ensure_root()
    target = resolve_target(name)
    if not target.is_file():
        raise UploadError(404, "file not found")
    try:
        target.unlink()
    except OSError as exc:
        raise UploadError(500, f"delete failed: {type(exc).__name__}") from exc
