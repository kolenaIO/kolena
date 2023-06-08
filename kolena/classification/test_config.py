# Copyright 2021-2023 Kolena Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from kolena._api.v1.detection import Metrics
from kolena.detection._internal import TestConfig as _TestConfig
from kolena.errors import InputValidationError


class TestConfig(_TestConfig):
    """
    Base class for testing configurations.

    See concrete implementations [`FixedGlobalThreshold`][kolena.classification.FixedGlobalThreshold] and
    [`AccuracyOptimal`][kolena.classification.AccuracyOptimal] for details.
    """


class FixedGlobalThreshold(TestConfig):
    """
    Test configuration that sets the default display threshold in Kolena to a fixed global threshold for all labels
    within the test suite.
    """

    fixed_threshold: float
    """The threshold used as the default when visualizing results in Kolena. Must be between 0 and 1."""

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
    Test configuration that sets the default display threshold in Kolena to the threshold that corresponds to the
    highest accuracy score across all images within the test suite being tested.

    This threshold is evaluated and set per label for test suites with multiple labels.
    """

    def __init__(self) -> None:
        ...

    def _to_run_config(self) -> Metrics.RunConfig:
        return Metrics.RunConfig(strategy=Metrics.RunStrategy.ACCURACY_OPTIMAL, iou_threshold=0)
