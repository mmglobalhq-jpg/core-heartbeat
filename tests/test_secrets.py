"""Reading secrets from mounted files instead of the environment.

The high-blast-radius secrets (three Supabase service-role keys, which bypass RLS,
and the JWT secret, which could forge logins) now arrive as read-only files and are
blanked in the container environment. This is not a privilege boundary — the only
docker-group member is the sole operator — it narrows *accidental* exposure:
`docker inspect`, `printenv`, and inheritance by child processes.

The empty-string rule below is the one that would break production if wrong.
"""

import pytest

from services import secrets
from services.secrets import secret


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    secrets.reset_cache()
    monkeypatch.delenv("TEST_KEY", raising=False)
    monkeypatch.delenv("TEST_KEY_FILE", raising=False)
    yield
    secrets.reset_cache()


def test_reads_from_the_file_when_pointed_at_one(tmp_path, monkeypatch):
    f = tmp_path / "k"; f.write_text("from-file\n")
    monkeypatch.setenv("TEST_KEY_FILE", str(f))
    assert secret("TEST_KEY") == "from-file"


def test_file_wins_over_environment(tmp_path, monkeypatch):
    f = tmp_path / "k"; f.write_text("from-file")
    monkeypatch.setenv("TEST_KEY_FILE", str(f))
    monkeypatch.setenv("TEST_KEY", "from-env")
    assert secret("TEST_KEY") == "from-file"


def test_falls_back_to_environment_when_no_file_configured(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "from-env")
    assert secret("TEST_KEY") == "from-env"


def test_an_EMPTY_env_var_counts_as_absent(tmp_path, monkeypatch):
    """The rule that keeps production working.

    Compose cannot UNSET a variable inherited from `env_file:` — only override it —
    so the compose file sets these to "". If empty won over the file, every
    Supabase call would authenticate with an empty key and fail.
    """
    f = tmp_path / "k"; f.write_text("from-file")
    monkeypatch.setenv("TEST_KEY_FILE", str(f))
    monkeypatch.setenv("TEST_KEY", "")
    assert secret("TEST_KEY") == "from-file"


def test_missing_file_degrades_to_env_rather_than_raising(tmp_path, monkeypatch):
    """A missing mount must not take the service down."""
    monkeypatch.setenv("TEST_KEY_FILE", str(tmp_path / "nope"))
    monkeypatch.setenv("TEST_KEY", "from-env")
    assert secret("TEST_KEY") == "from-env"


def test_empty_file_degrades_to_env(tmp_path, monkeypatch):
    f = tmp_path / "k"; f.write_text("   \n")
    monkeypatch.setenv("TEST_KEY_FILE", str(f))
    monkeypatch.setenv("TEST_KEY", "from-env")
    assert secret("TEST_KEY") == "from-env"


def test_returns_default_when_nothing_is_set():
    assert secret("TEST_KEY", "fallback") == "fallback"
    assert secret("TEST_KEY") is None


def test_trailing_newline_is_stripped(tmp_path, monkeypatch):
    """`echo secret > file` adds one; sending it to Supabase would 401."""
    f = tmp_path / "k"; f.write_text("abc123\n")
    monkeypatch.setenv("TEST_KEY_FILE", str(f))
    assert secret("TEST_KEY") == "abc123"


def test_file_is_cached_not_reread_per_call(tmp_path, monkeypatch):
    f = tmp_path / "k"; f.write_text("first")
    monkeypatch.setenv("TEST_KEY_FILE", str(f))
    assert secret("TEST_KEY") == "first"
    f.write_text("second")
    assert secret("TEST_KEY") == "first", "should be cached — no filesystem hit per call"
    secrets.reset_cache()
    assert secret("TEST_KEY") == "second"
