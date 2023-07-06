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
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from kolena._experimental.object_detection.evaluator_multiclass import MulticlassObjectDetectionEvaluator
from kolena._experimental.object_detection.evaluator_single_class import SingleClassObjectDetectionEvaluator
from kolena._experimental.object_detection.workflow import GroundTruth
from kolena._experimental.object_detection.workflow import Inference
from kolena._experimental.object_detection.workflow import TestCase
from kolena._experimental.object_detection.workflow import TestCaseMetrics
from kolena._experimental.object_detection.workflow import TestCaseMetricsSingleClass
from kolena._experimental.object_detection.workflow import TestSample
from kolena._experimental.object_detection.workflow import TestSampleMetrics
from kolena._experimental.object_detection.workflow import TestSampleMetricsSingleClass
from kolena._experimental.object_detection.workflow import TestSuite
from kolena._experimental.object_detection.workflow import TestSuiteMetrics
from kolena._experimental.object_detection.workflow import ThresholdConfiguration
from kolena.workflow import Evaluator
from kolena.workflow import Plot


class ObjectDetectionEvaluator(Evaluator):
    """
    This `ObjectDetectionEvaluator` transforms inferences into metrics for the object detection workflow for a
    single class or multiple classes. The `ObjectDetectionEvaluator` uses the [`SingleClassObjectDetectionEvaluator`]
    [kolena._experimental.object_detection.evaluator_single_class.SingleClassObjectDetectionEvaluator] for single
    class evaluation, and [`MulticlassObjectDetectionEvaluator`]
    [kolena._experimental.object_detection.evaluator_multiclass.MulticlassObjectDetectionEvaluator] for multiclass
    evaluation.

    When a [`ThresholdConfiguration`][kolena._experimental.object_detection.workflow.ThresholdConfiguration] is
    configured to use an F1-Optimal threshold strategy, the evaluator requires that the first test case retrieved for
    a test suite contains the complete sample set.

    For additional functionality, see the associated [base class documentation][kolena.workflow.evaluator.Evaluator].
    """

    # The evaluator class to use for single or multiclass object detection
    evaluator: Optional[
        Union[
            SingleClassObjectDetectionEvaluator,
            MulticlassObjectDetectionEvaluator,
        ]
    ] = None

    def compute_test_sample_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        configuration: Optional[ThresholdConfiguration] = None,
    ) -> List[Tuple[TestSample, Union[TestSampleMetrics, TestSampleMetricsSingleClass]]]:
        assert configuration is not None, "must specify configuration"

        # Use complete test case to determine workflow, single class or multiclass
        if self.evaluator is None:
            if configuration.with_class_level_metrics:
                self.evaluator = MulticlassObjectDetectionEvaluator()
            else:
                self.evaluator = SingleClassObjectDetectionEvaluator()

        return self.evaluator.compute_test_sample_metrics(
            test_case=test_case,
            inferences=inferences,
            configuration=configuration,
        )

    def compute_test_case_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        metrics: List[Union[TestSampleMetrics, TestSampleMetricsSingleClass]],
        configuration: Optional[ThresholdConfiguration] = None,
    ) -> Union[TestCaseMetrics, TestCaseMetricsSingleClass]:
        assert configuration is not None, "must specify configuration"
        return self.evaluator.compute_test_case_metrics(
            test_case=test_case,
            inferences=inferences,
            metrics=metrics,
            configuration=configuration,
        )

    def compute_test_case_plots(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        metrics: List[Union[TestSampleMetrics, TestSampleMetricsSingleClass]],
        configuration: Optional[ThresholdConfiguration] = None,
    ) -> Optional[List[Plot]]:
        assert configuration is not None, "must specify configuration"
        return self.evaluator.compute_test_case_plots(
            test_case=test_case,
            inferences=inferences,
            metrics=metrics,
            configuration=configuration,
        )

    def compute_test_suite_metrics(
        self,
        test_suite: TestSuite,
        metrics: List[Tuple[TestCase, Union[TestCaseMetrics, TestCaseMetricsSingleClass]]],
        configuration: Optional[ThresholdConfiguration] = None,
    ) -> TestSuiteMetrics:
        assert configuration is not None, "must specify configuration"
        return self.evaluator.compute_test_suite_metrics(
            test_suite=test_suite,
            metrics=metrics,
            configuration=configuration,
        )
