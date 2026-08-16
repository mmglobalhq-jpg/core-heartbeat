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

  {% if market and market.has_data %}
  <div style="margin-top:20px;padding:14px 16px;background:#ffffff;border:1px solid #e3e5e9;border-radius:8px;">
    {% if market.indices %}
    <div style="font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#5b6270;">
      Markets &middot; close {{ market.indices[0].as_of.strftime('%a %-d %b') }}
    </div>
    <table role="presentation" cellpadding="0" cellspacing="0" style="width:100%;margin-top:8px;font-size:14px;">
      {% for q in market.indices %}
      <tr>
        <td style="padding:3px 0;color:#2c3038;">{{ q.label }}</td>
        <td style="padding:3px 0;text-align:right;font-variant-numeric:tabular-nums;">{{ '{:,.2f}'.format(q.close) }}</td>
        <td style="padding:3px 0 3px 12px;text-align:right;font-variant-numeric:tabular-nums;
                   color:{{ '#1a7f4b' if q.change >= 0 else '#b3261e' }};">
          {{ '{:+,.2f}'.format(q.change) }}
        </td>
        <td style="padding:3px 0 3px 10px;text-align:right;font-variant-numeric:tabular-nums;
                   color:{{ '#1a7f4b' if q.change >= 0 else '#b3261e' }};">
          {{ '{:+.2f}%'.format(q.pct_change) }}
        </td>
      </tr>
      {% endfor %}
    </table>
    {% endif %}

    {% if market.yields %}
    <div style="font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#5b6270;
                margin-top:14px;padding-top:12px;border-top:1px solid #eceef1;">
      Treasuries &middot; {{ market.yields[0].as_of.strftime('%a %-d %b') }}
    </div>
    <table role="presentation" cellpadding="0" cellspacing="0" style="width:100%;margin-top:8px;font-size:14px;">
      {% for y in market.yields %}
      <tr>
        <td style="padding:3px 0;color:#2c3038;">{{ y.label }}</td>
        <td style="padding:3px 0;text-align:right;font-variant-numeric:tabular-nums;">{{ '{:.2f}%'.format(y.yield_pct) }}</td>
        <td style="padding:3px 0 3px 12px;text-align:right;font-variant-numeric:tabular-nums;
                   color:{{ '#1a7f4b' if y.change_bps >= 0 else '#b3261e' }};">
          {{ '{:+.0f} bps'.format(y.change_bps) }}
        </td>
      </tr>
      {% endfor %}
      {% if market.spreads %}
      <tr><td colspan="3" style="padding:8px 0 0;font-size:13px;color:#5b6270;
                                 font-variant-numeric:tabular-nums;">
        {% for sp in market.spreads %}{{ sp.label }} {{ '{:+.0f}'.format(sp.value_bps) }} bps{% if not loop.last %} &middot; {% endif %}{% endfor %}
      </td></tr>
      {% endif %}
    </table>
    {% endif %}
  </div>
  {% endif %}

  {% if sports and sports.has_data %}
  <div style="margin-top:20px;padding:14px 16px;background:#ffffff;border:1px solid #e3e5e9;border-radius:8px;">
    <div style="font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#5b6270;">
      Scores &middot; {{ sports.date.strftime('%a %-d %b') }}
    </div>
    {% for league, games in sports.leagues.items() %}
    <div style="margin-top:12px;">
      <div style="font-size:13px;font-weight:650;color:#2c3038;">{{ league }}</div>
      {% for g in games %}
      <div style="font-size:14px;color:#2c3038;padding:2px 0;">
        {{ g.line }}{% if g.featured %} <span style="color:#8a90a0;font-size:12px;">&middot; SEC</span>{% endif %}
      </div>
      {% endfor %}
    </div>
    {% endfor %}
  </div>
  {% endif %}

  {% if reports %}
  <div style="margin-top:20px;padding:14px 16px;background:#ffffff;border:1px solid #e3e5e9;border-radius:8px;">
    <div style="font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#5b6270;">
      Research
    </div>
    <ul style="margin:8px 0 0;padding-left:18px;font-size:14px;color:#2c3038;">
      {% for r in reports %}
      <li style="margin:0 0 6px;">
        <a href="{{ r.url }}" style="color:#16181d;">{{ r.title }}</a>
        <span style="color:#8a90a0;">&mdash; {{ r.issuer_code }}, {{ r.published_on.strftime('%-d %b') }}</span>
      </li>
      {% endfor %}
    </ul>
  </div>
  {% endif %}

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
{% if market and market.indices %}
MARKETS — close {{ market.indices[0].as_of.strftime('%a %-d %b') }}
{% for q in market.indices %}
  {{ '%-10s'|format(q.label) }} {{ '{:>12,.2f}'.format(q.close) }}  {{ '{:>+9,.2f}'.format(q.change) }}  {{ '{:>+7.2f}%'.format(q.pct_change) }}
{%- endfor %}
{% endif %}
{%- if market and market.yields %}
TREASURIES — {{ market.yields[0].as_of.strftime('%a %-d %b') }}
{% for y in market.yields %}
  {{ '%-10s'|format(y.label) }} {{ '{:>7.2f}%'.format(y.yield_pct) }}  {{ '{:>+5.0f} bps'.format(y.change_bps) }}
{%- endfor %}
{% if market.spreads %}
  {% for sp in market.spreads %}{{ sp.label }} {{ '{:+.0f}'.format(sp.value_bps) }} bps{% if not loop.last %} · {% endif %}{% endfor %}
{% endif %}
{% endif %}
{%- if sports and sports.has_data %}
SCORES — {{ sports.date.strftime('%a %-d %b') }}
{% for league, games in sports.leagues.items() %}
  {{ league }}
{%- for g in games %}
    {{ g.line }}{% if g.featured %}  (SEC){% endif %}
{%- endfor %}
{% endfor %}
{%- endif %}
{%- if reports %}
RESEARCH
{% for r in reports %}
  * {{ r.title }}
    {{ r.issuer_code }}, {{ r.published_on.strftime('%-d %b') }} — {{ r.url }}
{%- endfor %}
{% endif %}
TOP {{ top|length }}
{% for section in top %}
{{ section.rank }}. {{ section.headline }}
   {{ section.body }}
   {{ section.source_name or 'source' }} — {{ section.url }}
{% endfor %}
--
Assembled from {{ meta.sources_ok|default(0) }} sources.
"""


def _context(draft: BriefingDraft) -> dict:
    deep = draft.deep_dive
    return {
        "date_label": draft.briefing_date.strftime("%A, %-d %B %Y"),
        "top": draft.top,
        # The Deep Dive was removed on 2026-08-15. These stay in the context so
        # a historical briefing — which still has its deep_dive row — renders
        # unchanged if it is ever re-rendered, and so restoring the section is a
        # template change rather than a plumbing change.
        "deep_dive": deep,
        "deep_dive_paragraphs": [p.strip() for p in (deep.body if deep else "").split("\n") if p.strip()],
        "market": draft.market,
        "sports": draft.sports,
        "reports": draft.reports or [],
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
