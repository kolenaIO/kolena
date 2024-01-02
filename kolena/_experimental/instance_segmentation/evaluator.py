# Copyright 2021-2024 Kolena Inc.
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
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from kolena._experimental.instance_segmentation.workflow import EvaluatorConfiguration
from kolena._experimental.instance_segmentation.workflow import GroundTruth
from kolena._experimental.instance_segmentation.workflow import Inference
from kolena._experimental.instance_segmentation.workflow import TestCase
from kolena._experimental.instance_segmentation.workflow import TestSample
from kolena._experimental.instance_segmentation.workflow import TestSuite
from kolena._experimental.object_detection.evaluator import ObjectDetectionEvaluator
from kolena._experimental.object_detection.workflow import GroundTruth as ObjectGroundTruth
from kolena._experimental.object_detection.workflow import Inference as ObjectInference
from kolena._experimental.object_detection.workflow import MetricsTestCase
from kolena._experimental.object_detection.workflow import MetricsTestSample
from kolena._experimental.object_detection.workflow import MetricsTestSuite
from kolena._experimental.object_detection.workflow import TestSampleMetrics
from kolena._experimental.object_detection.workflow import TestSampleMetricsSingleClass
from kolena.workflow import Evaluator
from kolena.workflow import Plot


class InstanceSegmentationEvaluator(Evaluator):
    """
    This `InstanceSegmentationEvaluator` transforms inferences into metrics for the instance segmentation workflow for a
    single class or multiple classes.

    When a [`EvaluatorConfiguration`][kolena._experimental.instance_segmentation.workflow.EvaluatorConfiguration] is
    configured to use an F1-Optimal threshold strategy, the evaluator requires that the first test case retrieved for
    a test suite contains the complete sample set.

    For additional functionality, see the associated [base class documentation][kolena.workflow.evaluator.Evaluator].
    """

    def __init__(self, configurations: Optional[List[EvaluatorConfiguration]] = None):
        self.evaluator = ObjectDetectionEvaluator()
        super().__init__(configurations)

    def compute_test_sample_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        configuration: Optional[EvaluatorConfiguration] = None,
    ) -> List[Tuple[TestSample, Union[TestSampleMetrics, TestSampleMetricsSingleClass]]]:
        inferences = InstanceSegmentationEvaluator._convert_inferences(inferences)
        return self.evaluator.compute_test_sample_metrics(test_case, inferences, configuration)

    def compute_test_case_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        metrics: List[MetricsTestSample],
        configuration: Optional[EvaluatorConfiguration] = None,
    ) -> MetricsTestCase:
        inferences = InstanceSegmentationEvaluator._convert_inferences(inferences)
        return self.evaluator.compute_test_case_metrics(
            test_case,
            inferences,
            metrics,
            configuration,
        )

    def compute_test_case_plots(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        metrics: List[Union[TestSampleMetrics, TestSampleMetricsSingleClass]],
        configuration: Optional[EvaluatorConfiguration] = None,
    ) -> Optional[List[Plot]]:
        inferences = InstanceSegmentationEvaluator._convert_inferences(inferences)
        return self.evaluator.compute_test_case_plots(
            test_case,
            inferences,
            metrics,
            configuration,
        )

    def compute_test_suite_metrics(
        self,
        test_suite: TestSuite,
        metrics: List[Tuple[TestCase, MetricsTestCase]],
        configuration: Optional[EvaluatorConfiguration] = None,
    ) -> Optional[MetricsTestSuite]:
        return self.evaluator.compute_test_suite_metrics(test_suite, metrics, configuration)

    @staticmethod
    def _convert_inferences(
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
    ) -> List[Tuple[TestSample, ObjectGroundTruth, ObjectInference]]:
        return [
            (
                ts,
                ObjectGroundTruth(bboxes=gt.polygons, ignored_bboxes=gt.ignored_polygons),
                ObjectInference(bboxes=inf.polygons, ignored=inf.ignored),
            )
            for ts, gt, inf in inferences
        ]
