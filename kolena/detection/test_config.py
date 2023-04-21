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
    pass


class FixedGlobalThreshold(TestConfig):
    """
    Test configuration that sets the default display threshold in the Kolena UI to be a fixed global threshold for all
    label classes within the test run.
    """

    # The threshold used as the default for all label classes when visualizing results in the Kolena UI.
    # Must be between 0 and 1.
    fixed_threshold: float

    # The minimum intersection over union score between an inference and a ground truth for it to qualify as a potential
    # match.
    # Must be between 0 and 1.
    iou_threshold: float

    def __init__(self, fixed_threshold: float, iou_threshold: float = 0.5):
        if iou_threshold < 0 or iou_threshold > 1:
            raise InputValidationError(f"iou_threshold of {iou_threshold} was not between 0 and 1")
        if fixed_threshold < 0 or fixed_threshold > 1:
            raise InputValidationError(f"threshold of {fixed_threshold} was not between 0 and 1")

        self.iou_threshold = iou_threshold
        self.threshold = fixed_threshold

    def _to_run_config(self) -> Metrics.RunConfig:
        return Metrics.RunConfig(
            strategy=Metrics.RunStrategy.FIXED_GLOBAL_THRESHOLD,
            iou_threshold=self.iou_threshold,
            params=dict(threshold=self.threshold),
        )


class F1Optimal(TestConfig):
    """
    Test configuration that sets the default display threshold in the Kolena UI to be dynamically set to the threshold
    that corresponds to the highest F1 score for the test suite within the test run.

    This threshold is evaluated and set per label for test suites with multiple label classes.
    """

    # The minimum intersection over union score between an inference and a ground truth for it to qualify as a potential
    # match.
    # Must be between 0 and 1.
    iou_threshold: float

    def __init__(self, iou_threshold: float = 0.5):
        if iou_threshold < 0 or iou_threshold > 1:
            raise InputValidationError(f"iou_threshold of {iou_threshold} was not between 0 and 1")
        self.iou_threshold = iou_threshold

    def _to_run_config(self) -> Metrics.RunConfig:
        return Metrics.RunConfig(strategy=Metrics.RunStrategy.F1_OPTIMAL, iou_threshold=self.iou_threshold)
