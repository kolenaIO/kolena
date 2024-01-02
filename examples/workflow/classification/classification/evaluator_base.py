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
from abc import ABC
from abc import abstractmethod
from dataclasses import make_dataclass
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from classification.workflow import GroundTruth
from classification.workflow import Inference
from classification.workflow import TestCaseMetrics
from classification.workflow import TestCaseMetricsSingleClass
from classification.workflow import TestSampleMetrics
from classification.workflow import TestSampleMetricsSingleClass
from classification.workflow import TestSuiteMetrics
from classification.workflow import ThresholdConfiguration

from kolena._experimental.classification.utils import create_histogram
from kolena.workflow import Plot
from kolena.workflow.plot import Histogram


class BaseClassificationEvaluator(ABC):
    @abstractmethod
    def compute_test_sample_metrics(
        self,
        ground_truth: GroundTruth,
        inference: Inference,
        threshold_configuration: ThresholdConfiguration,
    ) -> Union[TestSampleMetricsSingleClass, TestSampleMetrics]:
        raise NotImplementedError

    @abstractmethod
    def compute_test_case_metrics(
        self,
        ground_truths: List[GroundTruth],
        metrics_test_samples: List[Union[TestSampleMetricsSingleClass, TestSampleMetrics]],
    ) -> Union[TestCaseMetricsSingleClass, TestCaseMetrics]:
        raise NotImplementedError

    @abstractmethod
    def compute_test_case_plots(
        self,
        ground_truths: List[GroundTruth],
        inferences: List[Inference],
        metrics: List[Union[TestSampleMetricsSingleClass, TestSampleMetrics]],
        gt_labels: List[str],
        confidence_range: Optional[Tuple[float, float, int]],
    ) -> List[Plot]:
        raise NotImplementedError

    def compute_test_suite_metrics(
        self,
        test_sample_metrics: List[TestSampleMetrics],
        configuration: ThresholdConfiguration,
    ) -> TestSuiteMetrics:
        n_images = len(test_sample_metrics)
        n_correct = sum([tsm.is_correct for tsm in test_sample_metrics])
        metrics = dict(
            n_images=n_images,
            n_invalid=len([mts for mts in test_sample_metrics if mts.classification is None]),
            n_correct=n_correct,
            overall_accuracy=n_correct / n_images if n_images > 0 else 0,
        )

        if configuration.threshold is not None:
            dc = make_dataclass(
                "ExtendedTestSuiteMetrics",
                bases=(TestSuiteMetrics,),
                fields=[("threshold", float)],
                frozen=True,
            )
            metrics_test_suite = dc(
                **metrics,
                threshold=configuration.threshold,
            )
        else:
            metrics_test_suite = TestSuiteMetrics(
                **metrics,
            )

        return metrics_test_suite

    def _compute_test_case_confidence_histograms(
        self,
        metrics: List[Union[TestSampleMetricsSingleClass, TestSampleMetrics]],
        range: Tuple[float, float, int],
    ) -> List[Histogram]:
        all = [mts.classification.score for mts in metrics if mts.classification]
        correct = [mts.classification.score for mts in metrics if mts.classification and mts.is_correct]
        incorrect = [mts.classification.score for mts in metrics if mts.classification and not mts.is_correct]

        plots = [
            create_histogram(
                values=all,
                range=range,
                title="Score Distribution (All)",
                x_label="Confidence",
                y_label="Count",
            ),
            create_histogram(
                values=correct,
                range=range,
                title="Score Distribution (Correct)",
                x_label="Confidence",
                y_label="Count",
            ),
            create_histogram(
                values=incorrect,
                range=range,
                title="Score Distribution (Incorrect)",
                x_label="Confidence",
                y_label="Count",
            ),
        ]
        return plots
