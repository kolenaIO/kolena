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
from collections import defaultdict
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import numpy as np

from kolena._experimental.object_detection import GroundTruth
from kolena._experimental.object_detection import Inference
from kolena._experimental.object_detection import TestCase
from kolena._experimental.object_detection import TestCaseMetricsSingleClass
from kolena._experimental.object_detection import TestSample
from kolena._experimental.object_detection import TestSampleMetrics
from kolena._experimental.object_detection import TestSampleMetricsSingleClass
from kolena._experimental.object_detection import TestSuite
from kolena._experimental.object_detection import TestSuiteMetrics
from kolena._experimental.object_detection import ThresholdConfiguration
from kolena._experimental.object_detection.utils import compute_average_precision
from kolena._experimental.object_detection.utils import compute_f1_plot
from kolena._experimental.object_detection.utils import compute_optimal_f1_threshold
from kolena._experimental.object_detection.utils import compute_pr_curve
from kolena._experimental.object_detection.utils import compute_pr_plot
from kolena._experimental.object_detection.utils import filter_inferences
from kolena.workflow import Evaluator
from kolena.workflow import EvaluatorConfiguration
from kolena.workflow import Plot
from kolena.workflow.metrics import f1_score as compute_f1_score
from kolena.workflow.metrics import InferenceMatches
from kolena.workflow.metrics import match_inferences
from kolena.workflow.metrics import precision as compute_precision
from kolena.workflow.metrics import recall as compute_recall


class SingleClassObjectDetectionEvaluator(Evaluator):
    """
    The `SingleClassObjectDetectionEvaluator` transforms inferences into metrics for the object detection workflow for
    a single class.

    When a [`ThresholdConfiguration`][kolena._experimental.object_detection.workflow.ThresholdConfiguration] is
    configured to use an F1-Optimal threshold strategy, the evaluator requires that the first test case retrieved for
    a test suite contains the complete sample set.

    For additional functionality, see the associated [base class documentation][kolena.workflow.evaluator.Evaluator].
    """

    threshold_cache: Dict[str, float]  # configuration -> threshold
    """
    Assumes that the first test case retrieved for the test suite contains the complete sample set to be used for
    F1-Optimal threshold computation. Subsequent requests for a given threshold strategy (for other test cases) will
    hit this cache and use the previously computed population level confidence thresholds.
    """

    locators_by_test_case: Dict[str, List[str]]
    """
    Keeps track of test sample locators for each test case (used for total # of image count in aggregated metrics).
    """

    matchings_by_test_case: Dict[str, Dict[str, List[InferenceMatches]]]
    """
    Caches matchings per configuration and test case for faster test case metric and plot computation.
    """

    def __init__(self, configurations: Optional[List[EvaluatorConfiguration]] = None):
        super().__init__(configurations)
        self.threshold_cache = {}
        self.locators_by_test_case = {}
        self.matchings_by_test_case = defaultdict(lambda: defaultdict(list))

    def test_sample_metrics_ignored(
        self,
        thresholds: float,
    ) -> TestSampleMetricsSingleClass:
        return TestSampleMetricsSingleClass(
            TP=[],
            FP=[],
            FN=[],
            count_TP=0,
            count_FP=0,
            count_FN=0,
            has_TP=False,
            has_FP=False,
            has_FN=False,
            ignored=True,
            max_confidence_above_t=None,
            min_confidence_above_t=None,
            thresholds=thresholds,
        )

    def test_sample_metrics_single_class(
        self,
        bbox_matches: InferenceMatches,
        thresholds: float,
    ) -> TestSampleMetricsSingleClass:
        tp = [inf for _, inf in bbox_matches.matched if inf.score >= thresholds]
        fp = [inf for inf in bbox_matches.unmatched_inf if inf.score >= thresholds]
        fn = bbox_matches.unmatched_gt + [gt for gt, inf in bbox_matches.matched if inf.score < thresholds]
        non_ignored_inferences = tp + fp
        scores = [inf.score for inf in non_ignored_inferences]
        return TestSampleMetricsSingleClass(
            TP=tp,
            FP=fp,
            FN=fn,
            count_TP=len(tp),
            count_FP=len(fp),
            count_FN=len(fn),
            has_TP=len(tp) > 0,
            has_FP=len(fp) > 0,
            has_FN=len(fn) > 0,
            ignored=False,
            max_confidence_above_t=max(scores) if len(scores) > 0 else None,
            min_confidence_above_t=min(scores) if len(scores) > 0 else None,
            thresholds=thresholds,
        )

    def compute_image_metrics(
        self,
        ground_truth: GroundTruth,
        inference: Inference,
        configuration: ThresholdConfiguration,
        test_case_name: str,
    ) -> TestSampleMetricsSingleClass:
        assert configuration is not None, "must specify configuration"
        thresholds = self.get_confidence_thresholds(configuration)
        if inference.ignored:
            return self.test_sample_metrics_ignored(thresholds)

        bbox_matches: InferenceMatches = match_inferences(
            ground_truth.bboxes,
            filter_inferences(inferences=inference.bboxes, confidence_score=configuration.min_confidence_score),
            ignored_ground_truths=ground_truth.ignored_bboxes,
            mode="pascal",
            iou_threshold=configuration.iou_threshold,
        )
        self.matchings_by_test_case[configuration.display_name()][test_case_name].append(bbox_matches)

        return self.test_sample_metrics_single_class(bbox_matches, thresholds)

    def compute_and_cache_f1_optimal_thresholds(
        self,
        configuration: ThresholdConfiguration,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
    ) -> None:
        if configuration.threshold_strategy != "F1-Optimal":
            return

        if configuration.display_name() in self.threshold_cache.keys():
            return

        all_bbox_matches = [
            match_inferences(
                ground_truth.bboxes,
                filter_inferences(inferences=inference.bboxes, confidence_score=configuration.min_confidence_score),
                ignored_ground_truths=ground_truth.ignored_bboxes,
                mode="pascal",
                iou_threshold=configuration.iou_threshold,
            )
            for _, ground_truth, inference in inferences
            if not inference.ignored
        ]
        optimal_thresholds = compute_optimal_f1_threshold(all_bbox_matches)
        self.threshold_cache[configuration.display_name()] = max(configuration.min_confidence_score, optimal_thresholds)

    def compute_test_sample_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],  # type: ignore
        configuration: Optional[ThresholdConfiguration] = None,  # type: ignore
    ) -> List[Tuple[TestSample, Union[TestSampleMetrics, TestSampleMetricsSingleClass]]]:
        assert configuration is not None, "must specify configuration"
        # compute thresholds to cache values for subsequent steps
        self.compute_and_cache_f1_optimal_thresholds(configuration, inferences)
        return [
            (ts, self.compute_image_metrics(gt, inf, configuration, test_case.name))  # type: ignore
            for ts, gt, inf in inferences
        ]

    def test_case_metrics_single_class(
        self,
        metrics: List[TestSampleMetricsSingleClass],
        average_precision: float,
    ) -> TestCaseMetricsSingleClass:
        tp_count = sum(im.count_TP for im in metrics)
        fp_count = sum(im.count_FP for im in metrics)
        fn_count = sum(im.count_FN for im in metrics)
        ignored_count = sum(1 if im.ignored else 0 for im in metrics)

        precision = compute_precision(tp_count, fp_count)
        recall = compute_recall(tp_count, fn_count)
        f1_score = compute_f1_score(tp_count, fp_count, fn_count)

        return TestCaseMetricsSingleClass(
            Objects=tp_count + fn_count,
            Inferences=tp_count + fp_count,
            TP=tp_count,
            FN=fn_count,
            FP=fp_count,
            nIgnored=ignored_count,
            Precision=precision,
            Recall=recall,
            F1=f1_score,
            AP=average_precision,
        )

    def compute_test_case_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],  # type: ignore
        metrics: List[TestSampleMetricsSingleClass],  # type: ignore
        configuration: Optional[ThresholdConfiguration] = None,  # type: ignore
    ) -> TestCaseMetricsSingleClass:
        assert configuration is not None, "must specify configuration"
        all_bbox_matches = self.matchings_by_test_case[configuration.display_name()][test_case.name]  # type: ignore
        self.locators_by_test_case[test_case.name] = [ts.locator for ts, _, _ in inferences]  # type: ignore

        average_precision = 0.0
        baseline_pr_curve = compute_pr_curve(all_bbox_matches)
        if baseline_pr_curve is not None:
            average_precision = compute_average_precision(baseline_pr_curve.y, baseline_pr_curve.x)  # type: ignore

        return self.test_case_metrics_single_class(metrics, average_precision)

    def compute_test_case_plots(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],  # type: ignore
        metrics: List[TestSampleMetricsSingleClass],  # type: ignore
        configuration: Optional[ThresholdConfiguration] = None,  # type: ignore
    ) -> Optional[List[Plot]]:
        assert configuration is not None, "must specify configuration"
        all_bbox_matches = self.matchings_by_test_case[configuration.display_name()][test_case.name]  # type: ignore

        plots: List[Plot] = []
        plots.extend(
            filter(
                None,
                [
                    compute_pr_plot(all_bbox_matches),
                    compute_f1_plot(all_bbox_matches),
                ],
            ),
        )

        return plots

    def test_suite_metrics(
        self,
        unique_locators: Set[str],
        average_precisions: List[float],
        threshold: Optional[float] = None,
    ) -> TestSuiteMetrics:
        return TestSuiteMetrics(
            n_images=len(unique_locators),
            mean_AP=np.mean(average_precisions) if average_precisions else 0.0,  # type: ignore
            threshold=threshold,
        )

    def compute_test_suite_metrics(
        self,
        test_suite: TestSuite,
        metrics: List[Tuple[TestCase, TestCaseMetricsSingleClass]],  # type: ignore
        configuration: Optional[ThresholdConfiguration] = None,  # type: ignore
    ) -> TestSuiteMetrics:
        assert configuration is not None, "must specify configuration"
        unique_locators = {
            locator for tc, _ in metrics for locator in self.locators_by_test_case[tc.name]  # type: ignore
        }
        average_precisions = [tcm.AP for _, tcm in metrics]
        threshold = self.get_confidence_thresholds(configuration)
        return self.test_suite_metrics(unique_locators, average_precisions, threshold)

    def get_confidence_thresholds(self, configuration: ThresholdConfiguration) -> float:
        if configuration.threshold_strategy == "F1-Optimal":
            return self.threshold_cache[configuration.display_name()]
        else:
            return configuration.threshold_strategy
