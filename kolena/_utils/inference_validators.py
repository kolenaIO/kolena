from pydantic import validate_arguments

from kolena._utils.validators import ValidatorConfig


@validate_arguments(config=ValidatorConfig)
def validate_label(label: str) -> None:
    if label.strip() == "":
        raise ValueError("label must contain non-whitespace characters", label)


@validate_arguments(config=ValidatorConfig)
def validate_confidence(confidence: float) -> None:
    if not (0 <= confidence <= 1):
        raise ValueError("confidence must be between 0 and 1 (inclusive)", confidence)
