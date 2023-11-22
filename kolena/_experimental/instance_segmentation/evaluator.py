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

from kolena._experimental.instance_segmentation import GroundTruth
from kolena._experimental.instance_segmentation import Inference
from kolena._experimental.instance_segmentation import TestCase
from kolena._experimental.instance_segmentation import TestCaseMetrics
from kolena._experimental.instance_segmentation import TestCaseMetricsSingleClass
from kolena._experimental.instance_segmentation import TestSample
from kolena._experimental.instance_segmentation import TestSampleMetrics
from kolena._experimental.instance_segmentation import TestSampleMetricsSingleClass
from kolena._experimental.instance_segmentation import TestSuite
from kolena._experimental.instance_segmentation import TestSuiteMetrics
from kolena._experimental.instance_segmentation import ThresholdConfiguration
from kolena._experimental.instance_segmentation.evaluator_multiclass import MulticlassInstanceSegmentationEvaluator
from kolena._experimental.instance_segmentation.evaluator_single_class import SingleClassInstanceSegmentationEvaluator
from kolena.workflow import Evaluator
from kolena.workflow import Plot


class InstanceSegmentationEvaluator(Evaluator):
    """
    This `InstanceSegmentationEvaluator` transforms inferences into metrics for the instance segmentation workflow for
    single class or multiple classes.

    When a [`ThresholdConfiguration`][kolena._experimental.instance_segmentation.workflow.ThresholdConfiguration] is
    configured to use an F1-Optimal threshold strategy, the evaluator requires that the first test case retrieved for
    a test suite contains the complete sample set.

    For additional functionality, see the associated [base class documentation][kolena.workflow.evaluator.Evaluator].
    """

    # The evaluator class to use for single or multiclass instance segmentation
    evaluator: Optional[
        Union[
            SingleClassInstanceSegmentationEvaluator,
            MulticlassInstanceSegmentationEvaluator,
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
            labels = {gt.label for _, gts, _ in inferences for gt in gts.polygons} | {
                inf.label for _, _, infs in inferences for inf in infs.polygons
            }
            if len(labels) >= 2:
                self.evaluator = MulticlassInstanceSegmentationEvaluator()
            else:
                self.evaluator = SingleClassInstanceSegmentationEvaluator()

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
