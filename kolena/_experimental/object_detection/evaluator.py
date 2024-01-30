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
# mypy: disable-error-code="override"
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from kolena._experimental.object_detection import GroundTruth
from kolena._experimental.object_detection import Inference
from kolena._experimental.object_detection import TestCase
from kolena._experimental.object_detection import TestCaseMetrics
from kolena._experimental.object_detection import TestCaseMetricsSingleClass
from kolena._experimental.object_detection import TestSample
from kolena._experimental.object_detection import TestSampleMetrics
from kolena._experimental.object_detection import TestSampleMetricsSingleClass
from kolena._experimental.object_detection import TestSuite
from kolena._experimental.object_detection import TestSuiteMetrics
from kolena._experimental.object_detection import ThresholdConfiguration
from kolena._experimental.object_detection.evaluator_multiclass import MulticlassObjectDetectionEvaluator
from kolena._experimental.object_detection.evaluator_single_class import SingleClassObjectDetectionEvaluator
from kolena.workflow import Evaluator
from kolena.workflow import EvaluatorConfiguration
from kolena.workflow import Plot


class ObjectDetectionEvaluator(Evaluator):
    """
    This `ObjectDetectionEvaluator` transforms inferences into metrics for the object detection workflow for a
    single class or multiple classes.

    When a [`ThresholdConfiguration`][kolena._experimental.object_detection.workflow.ThresholdConfiguration] is
    configured to use an F1-Optimal threshold strategy, the evaluator requires that the first test case retrieved for
    a test suite contains the complete sample set.

    For additional functionality, see the associated [base class documentation][kolena.workflow.evaluator.Evaluator].
    """

    def __init__(self, configurations: Optional[List[EvaluatorConfiguration]] = None):
        super().__init__(configurations)
        self.single_class_evaluator = SingleClassObjectDetectionEvaluator()
        self.multiclass_evaluator = MulticlassObjectDetectionEvaluator()
        self.dynamic_evaluator: Union[
            SingleClassObjectDetectionEvaluator,
            MulticlassObjectDetectionEvaluator,
            None,
        ] = None

    def compute_test_sample_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],  # type: ignore
        configuration: Optional[ThresholdConfiguration] = None,  # type: ignore
    ) -> List[Tuple[TestSample, Union[TestSampleMetrics, TestSampleMetricsSingleClass]]]:
        assert configuration is not None, "must specify configuration"

        if configuration.multiclass is None:
            labels = {gt.label for _, gts, _ in inferences for gt in gts.bboxes} | {
                inf.label for _, _, infs in inferences for inf in infs.bboxes
            }
            if len(labels) >= 2:
                self.dynamic_evaluator = self.multiclass_evaluator
            else:
                self.dynamic_evaluator = self.single_class_evaluator

        evaluator = self._get_evaluator(configuration)
        return evaluator.compute_test_sample_metrics(  # type: ignore
            test_case=test_case,
            inferences=inferences,
            configuration=configuration,
        )

    def compute_test_case_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],  # type: ignore
        metrics: List[Union[TestSampleMetrics, TestSampleMetricsSingleClass]],  # type: ignore
        configuration: Optional[ThresholdConfiguration] = None,  # type: ignore
    ) -> Union[TestCaseMetrics, TestCaseMetricsSingleClass]:
        assert configuration is not None, "must specify configuration"

        evaluator = self._get_evaluator(configuration)
        return evaluator.compute_test_case_metrics(  # type: ignore
            test_case=test_case,
            inferences=inferences,
            metrics=metrics,  # type: ignore
            configuration=configuration,
        )

    def compute_test_case_plots(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],  # type: ignore
        metrics: List[Union[TestSampleMetrics, TestSampleMetricsSingleClass]],  # type: ignore
        configuration: Optional[ThresholdConfiguration] = None,  # type: ignore
    ) -> Optional[List[Plot]]:
        assert configuration is not None, "must specify configuration"

        evaluator = self._get_evaluator(configuration)
        return evaluator.compute_test_case_plots(  # type: ignore
            test_case=test_case,
            inferences=inferences,
            metrics=metrics,  # type: ignore
            configuration=configuration,
        )

    def compute_test_suite_metrics(
        self,
        test_suite: TestSuite,
        metrics: List[Tuple[TestCase, Union[TestCaseMetrics, TestCaseMetricsSingleClass]]],  # type: ignore
        configuration: Optional[ThresholdConfiguration] = None,  # type: ignore
    ) -> TestSuiteMetrics:
        assert configuration is not None, "must specify configuration"

        evaluator = self._get_evaluator(configuration)
        return evaluator.compute_test_suite_metrics(  # type: ignore
            test_suite=test_suite,
            metrics=metrics,  # type: ignore
            configuration=configuration,
        )

    def _get_evaluator(
        self,
        configuration: Optional[ThresholdConfiguration],
    ) -> Union[SingleClassObjectDetectionEvaluator, MulticlassObjectDetectionEvaluator, None]:
        assert configuration is not None, "must specify configuration"

        if configuration.multiclass is None:
            return self.dynamic_evaluator
        return self.multiclass_evaluator if configuration.multiclass else self.single_class_evaluator
