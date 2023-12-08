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
from kolena._experimental.object_detection.evaluator_multiclass import MulticlassObjectDetectionEvaluator


class MulticlassInstanceSegmentationEvaluator(MulticlassObjectDetectionEvaluator):
    """
    The `MulticlassInstanceSegmentationEvaluator` transforms inferences into metrics for the
    Instance Segmentation workflow for multiple classes.

    When a [`ThresholdConfiguration`][kolena._experimental.object_detection.workflow.ThresholdConfiguration] is
    configured to use an F1-Optimal threshold strategy, the evaluator requires that the first test case retrieved for
    a test suite contains the complete sample set.

    For additional functionality, see the associated [base class documentation][kolena.workflow.evaluator.Evaluator].
    """
