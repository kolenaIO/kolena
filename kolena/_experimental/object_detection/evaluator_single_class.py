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
from collections import defaultdict
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import numpy as np

from kolena._experimental.object_detection import GroundTruth
from kolena._experimental.object_detection import Inference
from kolena._experimental.object_detection import TestCase
from kolena._experimental.object_detection import TestCaseMetricsSingleClass
from kolena._experimental.object_detection import TestSample
from kolena._experimental.object_detection import TestSampleMetricsSingleClass
from kolena._experimental.object_detection import TestSuite
from kolena._experimental.object_detection import TestSuiteMetrics
from kolena._experimental.object_detection import ThresholdConfiguration
from kolena._experimental.object_detection.utils import compute_average_precision
from kolena._experimental.object_detection.utils import compute_f1_plot
from kolena._experimental.object_detection.utils import compute_pr_curve
from kolena._experimental.object_detection.utils import compute_pr_plot
from kolena._experimental.object_detection.utils import filter_inferences
from kolena._experimental.object_detection.utils import get_thresholds
from kolena._experimental.object_detection.workflow import TestCaseThresholdedMetricsSingleClass
from kolena._experimental.object_detection.workflow import TestSampleThresholdedMetricsSingleClass
from kolena.workflow import Evaluator
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

    threshold_cache: Dict[str, float] = {}  # configuration -> threshold
    """
    Assumes that the first test case retrieved for the test suite contains the complete sample set to be used for
    F1-Optimal threshold computation. Subsequent requests for a given threshold strategy (for other test cases) will
    hit this cache and use the previously computed population level confidence thresholds.
    """

    locators_by_test_case: Dict[str, List[str]] = {}
    """
    Keeps track of test sample locators for each test case (used for total # of image count in aggregated metrics).
    """

    matchings_by_test_case: Dict[str, Dict[str, List[InferenceMatches]]] = defaultdict(
        lambda: defaultdict(list),
    )
    """
    Caches matchings per configuration and test case for faster test case metric and plot computation.
    """

    def test_sample_metrics_ignored(
        self,
    ) -> TestSampleMetricsSingleClass:
        return TestSampleMetricsSingleClass(
            ignored=True,
            Thresholded=[],
        )

    def test_sample_metrics_single_class(
        self,
        bbox_matches: InferenceMatches,
        thresholds: List[float],
    ) -> TestSampleMetricsSingleClass:
        thresholded_metrics = []
        for threshold in thresholds:
            tp = [inf for _, inf in bbox_matches.matched if inf.score >= thresholds]
            fp = [inf for inf in bbox_matches.unmatched_inf if inf.score >= thresholds]
            fn = bbox_matches.unmatched_gt + [gt for gt, inf in bbox_matches.matched if inf.score < thresholds]
            non_ignored_inferences = tp + fp
            scores = [inf.score for inf in non_ignored_inferences]
            thresholded_metrics.append(
                TestSampleThresholdedMetricsSingleClass(
                    threshold=threshold,
                    TP=tp,
                    FP=fp,
                    FN=fn,
                    count_TP=len(tp),
                    count_FP=len(fp),
                    count_FN=len(fn),
                    has_TP=len(tp) > 0,
                    has_FP=len(fp) > 0,
                    has_FN=len(fn) > 0,
                    max_confidence_above_t=max(scores) if len(scores) > 0 else None,
                    min_confidence_above_t=min(scores) if len(scores) > 0 else None,
                ),
            )
        return TestSampleMetricsSingleClass(
            ignored=False,
            Thresholded=thresholded_metrics,
        )

    def compute_image_metrics(
        self,
        ground_truth: GroundTruth,
        inference: Inference,
        configuration: ThresholdConfiguration,
        test_case_name: str,
    ) -> TestSampleMetricsSingleClass:
        assert configuration is not None, "must specify configuration"
        if inference.ignored:
            return self.test_sample_metrics_ignored()

        bbox_matches: InferenceMatches = match_inferences(
            ground_truth.bboxes,
            filter_inferences(inferences=inference.bboxes, confidence_score=configuration.min_confidence_score),
            ignored_ground_truths=ground_truth.ignored_bboxes,
            mode="pascal",
            iou_threshold=configuration.iou_threshold,
        )
        self.matchings_by_test_case[configuration.display_name()][test_case_name].append(bbox_matches)

        return self.test_sample_metrics_single_class(bbox_matches, get_thresholds())

    def compute_test_sample_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        configuration: Optional[ThresholdConfiguration] = None,
    ) -> List[Tuple[TestSample, TestSampleMetricsSingleClass]]:
        assert configuration is not None, "must specify configuration"
        return [(ts, self.compute_image_metrics(gt, inf, configuration, test_case.name)) for ts, gt, inf in inferences]

    def test_case_metrics_single_class(
        self,
        thresholds: List[float],
        metrics: List[TestSampleMetricsSingleClass],
        average_precision: float,
    ) -> TestCaseMetricsSingleClass:
        thresholded_metrics = []
        for i, threshold in enumerate(thresholds):
            tp_count = sum(im.object[i].count_TP for im in metrics)
            fp_count = sum(im.object[i].count_FP for im in metrics)
            fn_count = sum(im.object[i].count_FN for im in metrics)

            precision = compute_precision(tp_count, fp_count)
            recall = compute_recall(tp_count, fn_count)
            f1_score = compute_f1_score(tp_count, fp_count, fn_count)

            thresholded_metrics.append(
                TestCaseThresholdedMetricsSingleClass(
                    threshold=threshold,
                    Objects=tp_count + fn_count,
                    Inferences=tp_count + fp_count,
                    TP=tp_count,
                    FN=fn_count,
                    FP=fp_count,
                    Precision=precision,
                    Recall=recall,
                    F1=f1_score,
                ),
            )
        ignored_count = sum(1 if im.ignored else 0 for im in metrics)

        return TestCaseMetricsSingleClass(
            ClassSpecific=thresholded_metrics,
            nIgnored=ignored_count,
            AP=average_precision,
        )

    def compute_test_case_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        metrics: List[TestSampleMetricsSingleClass],
        configuration: Optional[ThresholdConfiguration] = None,
    ) -> TestCaseMetricsSingleClass:
        assert configuration is not None, "must specify configuration"
        all_bbox_matches = self.matchings_by_test_case[configuration.display_name()][test_case.name]
        self.locators_by_test_case[test_case.name] = [ts.locator for ts, _, _ in inferences]

        average_precision = 0.0
        baseline_pr_curve = compute_pr_curve(all_bbox_matches)
        if baseline_pr_curve is not None:
            average_precision = compute_average_precision(baseline_pr_curve.y, baseline_pr_curve.x)

        return self.test_case_metrics_single_class(get_thresholds(), metrics, average_precision)

    def compute_test_case_plots(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        metrics: List[TestSampleMetricsSingleClass],
        configuration: Optional[ThresholdConfiguration] = None,
    ) -> Optional[List[Plot]]:
        assert configuration is not None, "must specify configuration"
        all_bbox_matches = self.matchings_by_test_case[configuration.display_name()][test_case.name]

        plots: Optional[List[Plot]] = []
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

    def test_suite_metrics(self, unique_locators: Set[str], average_precisions: List[float]) -> TestSuiteMetrics:
        return TestSuiteMetrics(
            n_images=len(unique_locators),
            mean_AP=np.mean(average_precisions) if average_precisions else 0.0,
        )

    def compute_test_suite_metrics(
        self,
        test_suite: TestSuite,
        metrics: List[Tuple[TestCase, TestCaseMetricsSingleClass]],
        configuration: Optional[ThresholdConfiguration] = None,
    ) -> TestSuiteMetrics:
        assert configuration is not None, "must specify configuration"
        unique_locators = {locator for tc, _ in metrics for locator in self.locators_by_test_case[tc.name]}
        average_precisions = [tcm.AP for _, tcm in metrics]
        return self.test_suite_metrics(unique_locators, average_precisions)

    def get_confidence_thresholds(self, configuration: ThresholdConfiguration) -> float:
        if configuration.threshold_strategy == "F1-Optimal":
            return self.threshold_cache[configuration.display_name()]
        else:
            return configuration.threshold_strategy
