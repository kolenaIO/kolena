import pytest

from kolena._utils.inference_validators import validate_confidence
from kolena._utils.inference_validators import validate_label


@pytest.mark.parametrize(
    "label, is_valid",
    [
        ("label", True),
        ("label with spaces", True),
        ("", False),
        ("   ", False),
        (" \n\r  ", False),
    ],
)
def test_validate_label(label: str, is_valid: bool) -> None:
    if not is_valid:
        with pytest.raises(ValueError):
            validate_label(label)
    else:
        validate_label(label)


@pytest.mark.parametrize(
    "confidence, is_valid",
    [
        (0, True),
        (1, True),
        (1 / 3, True),
        (0.987654321, True),
        (-0, True),
        (-1, False),
        (2, False),
        (None, False),
        (float("nan"), False),
        (float("inf"), False),
        (-float("inf"), False),
    ],
)
def test_validate_confidence(confidence: float, is_valid: bool) -> None:
    if not is_valid:
        with pytest.raises(ValueError):
            validate_confidence(confidence)
    else:
        validate_confidence(confidence)
