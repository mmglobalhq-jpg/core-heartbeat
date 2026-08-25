"""The composer must not ask permission to READ, and must still confirm WRITES.

The reported defect: asked to find flights, the assistant replied "To do this, I
will need to search the web. Would you like me to do that?" and spent a turn
waiting for "yes". The cause was capabilities_block advertising every tool "on the
user's confirmation" — wording written for calendar writes and applied to reads.

The risk in fixing it is the opposite failure, which this platform has already had:
a confirmation step that stops working means writes happen unannounced. So both
halves are asserted together, in the one file, deliberately.
"""
import re

import pytest

from orchestrator import capabilities_block, _build_local_prompt, _tool_catalogue_block
from models import IntentPayload


def _prompt(raw: str = "find me a flight to Memphis") -> str:
    payload = IntentPayload(intent="chat", confidence=0.9, entities={}, raw_input=raw, source="test")
    return _build_local_prompt({"intent": payload, "messages": [], "prior_context": []})


# --- reads: no permission ---------------------------------------------------


def test_capabilities_no_longer_gate_every_tool_on_confirmation():
    block = capabilities_block()
    assert "on the user's confirmation" not in block
    assert "What this assistant can do (via its tools):" in block


def test_reads_are_explicitly_exempt_from_asking():
    block = capabilities_block().lower()
    assert "a read" in block and "needs no permission" in block
    # the exact phrasing the user was served, called out by name
    assert "would you like me to search?" in block


def test_composer_prompt_carries_the_read_exemption():
    assert "needs NO permission" in _prompt()


# --- writes: still confirmed ------------------------------------------------


def test_writes_are_still_confirmed_first():
    block = capabilities_block()
    assert "A CHANGE" in block
    assert "Want me to go ahead?" in block
    assert "the next turn performs it" in block


def test_the_never_claim_an_action_happened_rule_survives():
    # Production once wrote 9 calendar events and then apologised for not writing
    # them; this paragraph is why. Narrowing the confirmation rule must not touch it.
    block = capabilities_block()
    assert "You do NOT run tools in this step" in block
    for forbidden in ("proceeding to", "I'm adding", "adding now", "done", "added", "scheduled"):
        assert forbidden in block


# --- search_web honesty -----------------------------------------------------


def test_web_family_note_is_no_longer_empty():
    # It was "" while every other family carried guidance — the one tool whose
    # limits the composer most needed to know had none stated.
    note = _tool_catalogue_block()
    assert "search_web returns a PROSE SUMMARY" in note


def test_composer_is_told_not_to_substitute_adjacent_information():
    note = _tool_catalogue_block()
    assert "SAY THAT PLAINLY" in note
    assert "drive times" in note  # the specific substitution that was served
    assert "live flight schedules" in note


# --- formatting -------------------------------------------------------------


def test_composer_is_told_the_output_is_markdown():
    p = _prompt()
    assert "rendered as Markdown" in p
    assert "one bullet per option" in p


def test_formatting_rules_forbid_the_filler_that_was_reported():
    p = _prompt()
    assert "let me know if you need anything else" in p
    assert "Prose answers stay prose" in p


@pytest.mark.parametrize("phrase", ["**bold**", "Bold sparingly"])
def test_bold_guidance_present(phrase):
    assert phrase in _prompt()
