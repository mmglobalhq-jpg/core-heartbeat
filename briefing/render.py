"""Render a briefing as HTML and as plain text.

EVERYTHING RENDERED HERE ORIGINATED ON THE PUBLIC WEB, so autoescaping is on and
is not optional. A headline is attacker-controlled text; a template that
interpolated it raw would turn an injected ``<script>`` — or, more realistically,
a broken tag that swallows the rest of the email — into a rendering bug at best.

Email HTML is written the boring way on purpose: a table-free single column,
inline styles, no external assets, no web fonts, no media queries doing anything
load-bearing. Mail clients are a hostile rendering target and the fashionable
techniques are exactly the ones Outlook and Gmail strip.

Dark mode is handled with ``prefers-color-scheme`` and explicit light colours as
the base, because a client that ignores the media query must still produce
readable output rather than black-on-black.
"""

from __future__ import annotations

import datetime as dt

from jinja2 import Environment, select_autoescape

from briefing.models import BriefingDraft

_env = Environment(autoescape=select_autoescape(default_for_string=True, default=True))


HTML_TEMPLATE = """\
<div style="margin:0;padding:0;background:#f4f5f7;">
<div style="max-width:640px;margin:0 auto;padding:24px 16px;
            font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
            color:#16181d;line-height:1.55;">

  <div style="padding-bottom:16px;border-bottom:2px solid #16181d;">
    <div style="font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:#5b6270;">
      Daily Briefing
    </div>
    <div style="font-size:22px;font-weight:700;margin-top:4px;">{{ date_label }}</div>
  </div>

  <div style="font-size:12px;letter-spacing:.08em;text-transform:uppercase;
              color:#5b6270;margin:24px 0 8px;">Top {{ top|length }}</div>

  {% for section in top %}
  <div style="padding:14px 0;border-bottom:1px solid #e3e5e9;">
    <div style="font-size:17px;font-weight:650;margin-bottom:6px;">
      <span style="color:#8a90a0;">{{ section.rank }}.</span>
      <a href="{{ section.url }}" style="color:#16181d;text-decoration:none;">{{ section.headline }}</a>
    </div>
    <div style="font-size:15px;color:#2c3038;">{{ section.body }}</div>
    <div style="font-size:12px;color:#8a90a0;margin-top:6px;">
      {{ section.source_name or 'source' }} &middot;
      <a href="{{ section.url }}" style="color:#5b6270;">read it</a>
    </div>
  </div>
  {% endfor %}

  {% if deep_dive %}
  <div style="margin-top:28px;padding:18px;background:#ffffff;border:1px solid #e3e5e9;border-radius:8px;">
    <div style="font-size:12px;letter-spacing:.08em;text-transform:uppercase;
                color:#5b6270;margin-bottom:8px;">Deep Dive</div>
    <div style="font-size:18px;font-weight:650;margin-bottom:10px;">
      <a href="{{ deep_dive.url }}" style="color:#16181d;text-decoration:none;">{{ deep_dive.headline }}</a>
    </div>
    {% for para in deep_dive_paragraphs %}
    <p style="font-size:15px;color:#2c3038;margin:0 0 12px;">{{ para }}</p>
    {% endfor %}
    <div style="font-size:12px;color:#8a90a0;">{{ deep_dive.source_name or 'source' }}</div>
  </div>
  {% endif %}

  <div style="margin-top:28px;font-size:11px;color:#8a90a0;line-height:1.5;">
    Assembled from {{ meta.sources_ok|default(0) }} sources.
    {%- if meta.blocked %} {{ meta.blocked }} item(s) could not be read and were summarised from
    headlines only.{% endif %}
    Every item links to its source.
  </div>

</div>
</div>
"""


TEXT_TEMPLATE = """\
DAILY BRIEFING — {{ date_label }}

TOP {{ top|length }}
{% for section in top %}
{{ section.rank }}. {{ section.headline }}
   {{ section.body }}
   {{ section.source_name or 'source' }} — {{ section.url }}
{% endfor %}
{% if deep_dive %}
DEEP DIVE — {{ deep_dive.headline }}

{{ deep_dive.body }}

{{ deep_dive.source_name or 'source' }} — {{ deep_dive.url }}
{% endif %}
--
Assembled from {{ meta.sources_ok|default(0) }} sources.
"""


def _context(draft: BriefingDraft) -> dict:
    deep = draft.deep_dive
    return {
        "date_label": draft.briefing_date.strftime("%A, %-d %B %Y"),
        "top": draft.top,
        "deep_dive": deep,
        # Split for HTML only. The plain-text version keeps the original blank
        # lines, so it does not need this.
        "deep_dive_paragraphs": [p.strip() for p in (deep.body if deep else "").split("\n") if p.strip()],
        "meta": draft.run_meta,
    }


def render_html(draft: BriefingDraft) -> str:
    return _env.from_string(HTML_TEMPLATE).render(**_context(draft))


def render_text(draft: BriefingDraft) -> str:
    # Autoescaping is for HTML; escaping &amp; into a text/plain body would be a
    # bug, so this template is rendered without it.
    plain = Environment(autoescape=False)  # noqa: S701 — text/plain, not markup
    return plain.from_string(TEXT_TEMPLATE).render(**_context(draft))


def subject_line(draft: BriefingDraft) -> str:
    lead = draft.top[0].headline if draft.top else "Your briefing"
    day = draft.briefing_date.strftime("%a %-d %b")
    return f"Briefing, {day}: {lead[:80]}"


def render_all(draft: BriefingDraft) -> dict[str, str]:
    return {
        "subject": subject_line(draft),
        "html": render_html(draft),
        "text": render_text(draft),
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
    }
