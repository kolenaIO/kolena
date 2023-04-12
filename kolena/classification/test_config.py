from kolena._api.v1.detection import Metrics
from kolena.detection._internal import TestConfig as _TestConfig
from kolena.errors import InputValidationError


class TestConfig(_TestConfig):
    ...


class FixedGlobalThreshold(TestConfig):
    """
    Test configuration that sets the default display threshold in the Kolena UI to be a fixed global threshold for all
    label classes within the test run.
    """

    # The threshold used as the default when visualizing results in the Kolena UI.
    # Must be between 0 and 1.
    fixed_threshold: float

    def __init__(self, fixed_threshold: float):
        if fixed_threshold < 0 or fixed_threshold > 1:
            raise InputValidationError(f"threshold of {fixed_threshold} was not between 0 and 1")
        self.fixed_threshold = fixed_threshold

    def _to_run_config(self) -> Metrics.RunConfig:
        return Metrics.RunConfig(
            strategy=Metrics.RunStrategy.FIXED_GLOBAL_THRESHOLD,
            iou_threshold=0,
            params=dict(threshold=self.fixed_threshold),
        )


class AccuracyOptimal(TestConfig):
    """
    Test configuration that sets the default display threshold in the Kolena UI to be dynamically set to the threshold
    that corresponds to the highest accuracy score for the test suite within the test run.

    This threshold is evaluated and set per label for test suites with multiple label classes.
    """

    def __init__(self) -> None:
        ...

    def _to_run_config(self) -> Metrics.RunConfig:
        return Metrics.RunConfig(strategy=Metrics.RunStrategy.ACCURACY_OPTIMAL, iou_threshold=0)
