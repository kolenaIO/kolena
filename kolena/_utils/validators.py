from pydantic import Extra


class ValidatorConfig:
    """Pydantic configuration for dataclasses and @validate_arguments decorators."""

    arbitrary_types_allowed = True
    smart_union = True
    extra = Extra.allow  # do not fail when unrecognized values are provided
