"""Pure-logic tests for the gateway: threshold decision and config loader.

No HTTP layer — exercises router.decide and router.load_confidence_threshold
directly (SC-002, SC-005, config edge cases / Scenario 7).
"""

import pytest

from router import DEFAULT_THRESHOLD, THRESHOLD_ENV_VAR, decide, load_confidence_threshold


# --- decide() : inclusive >= boundary ---------------------------------------

@pytest.mark.parametrize(
    "confidence,threshold,expected",
    [
        (0.9, 0.5, True),    # above
        (0.5, 0.5, True),    # exactly at (inclusive)
        (0.49, 0.5, False),  # below
        (0.0, 0.0, True),    # both at floor
        (1.0, 1.0, True),    # both at ceiling
        (0.0, 0.5, False),
    ],
)
def test_decide(confidence, threshold, expected):
    assert decide(confidence, threshold) is expected


# --- load_confidence_threshold() : env parsing ------------------------------

def test_threshold_unset_uses_default():
    assert load_confidence_threshold(env={}) == DEFAULT_THRESHOLD


def test_threshold_blank_uses_default():
    assert load_confidence_threshold(env={THRESHOLD_ENV_VAR: "   "}) == DEFAULT_THRESHOLD


@pytest.mark.parametrize("raw,value", [("0.0", 0.0), ("0.5", 0.5), ("1.0", 1.0), ("0.75", 0.75)])
def test_threshold_valid_in_range(raw, value):
    assert load_confidence_threshold(env={THRESHOLD_ENV_VAR: raw}) == value


@pytest.mark.parametrize("raw", ["-0.1", "1.1", "2", "-5"])
def test_threshold_out_of_range_raises(raw):
    with pytest.raises(ValueError) as exc:
        load_confidence_threshold(env={THRESHOLD_ENV_VAR: raw})
    assert THRESHOLD_ENV_VAR in str(exc.value)


@pytest.mark.parametrize("raw", ["abc", "", "0.5x", "high"])
def test_threshold_unparseable_raises_or_defaults(raw):
    # Blank is a valid "use default" case; non-numeric non-blank must raise.
    if raw.strip() == "":
        assert load_confidence_threshold(env={THRESHOLD_ENV_VAR: raw}) == DEFAULT_THRESHOLD
    else:
        with pytest.raises(ValueError) as exc:
            load_confidence_threshold(env={THRESHOLD_ENV_VAR: raw})
        assert THRESHOLD_ENV_VAR in str(exc.value)
