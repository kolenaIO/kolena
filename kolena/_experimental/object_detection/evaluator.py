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
from dataclasses import make_dataclass
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import numpy as np

from kolena._experimental.object_detection.utils import compute_ap
from kolena._experimental.object_detection.utils import compute_confusion_matrix_plot
from kolena._experimental.object_detection.utils import compute_optimal_f1
from kolena._experimental.object_detection.utils import compute_pr_f1_plots
from kolena._experimental.object_detection.utils import compute_pr_plot
from kolena._experimental.object_detection.utils import threshold_key
from kolena._experimental.object_detection.workflow import ClassMetricsPerTestCase
from kolena._experimental.object_detection.workflow import GroundTruth
from kolena._experimental.object_detection.workflow import Inference
from kolena._experimental.object_detection.workflow import TestCase
from kolena._experimental.object_detection.workflow import TestCaseMetrics
from kolena._experimental.object_detection.workflow import TestSample
from kolena._experimental.object_detection.workflow import TestSampleMetrics
from kolena._experimental.object_detection.workflow import TestSuite
from kolena._experimental.object_detection.workflow import TestSuiteMetrics
from kolena._experimental.object_detection.workflow import ThresholdConfiguration
from kolena._experimental.object_detection.workflow import ThresholdStrategy
from kolena.workflow import Evaluator
from kolena.workflow import Plot
from kolena.workflow.metrics import match_inferences_multiclass
from kolena.workflow.metrics._geometry import MulticlassInferenceMatches


class ObjectDetectionEvaluator(Evaluator):
    # Assumes that the first test case retrieved for the test suite contains the complete sample set to be used for
    # F1-Optimal threshold computation. Subsequent requests for a given threshold strategy (for other test cases) will
    # hit this cache and use the previously computed population level confidence thresholds.
    threshold_cache: Dict[str, Dict[str, float]] = {}  # configuration -> label -> threshold

    # Keeps track of test sample locators for each test case (used for total # of image count in aggregated metrics)
    locators_by_test_case: Dict[str, List[str]] = {}

    def compute_image_metrics(
        self,
        ground_truth: GroundTruth,
        inference: Inference,
        configuration: ThresholdConfiguration,
        labels: Set[str],  # the labels being tested in this test case
    ) -> TestSampleMetrics:
        thresholds = self.get_confidence_thresholds(configuration)
        bbox_matches: MulticlassInferenceMatches = match_inferences_multiclass(
            ground_truth.bboxes,
            [inf for inf in inference.bboxes if inf.score >= configuration.min_confidence_score],
            ignored_ground_truths=ground_truth.ignored_bboxes,
            mode="pascal",
            iou_threshold=configuration.iou_threshold,
        )
        tp = [inf for _, inf in bbox_matches.matched if inf.score >= thresholds[inf.label] and inf.label in labels]
        fp = [inf for inf in bbox_matches.unmatched_inf if inf.score >= thresholds[inf.label] and inf.label in labels]
        fn = [gt for gt, _ in bbox_matches.unmatched_gt if gt.label in labels] + [
            gt for gt, inf in bbox_matches.matched if inf.score < thresholds[inf.label] and gt.label in labels
        ]
        confused = [
            inf
            for gt, inf in bbox_matches.unmatched_gt
            if inf is not None and inf.score >= thresholds[inf.label] and gt.label in labels
        ]
        non_ignored_inferences = [inf for _, inf in bbox_matches.matched] + bbox_matches.unmatched_inf
        scores = [
            inf.score for inf in non_ignored_inferences if inf.score >= thresholds[inf.label] and inf.label in labels
        ]
        fields = [(threshold_key(label), float) for label in thresholds.keys()]
        dc = make_dataclass("ExtendedImageMetrics", bases=(TestSampleMetrics,), fields=fields, frozen=True)
        return dc(
            TP=tp,
            FP=fp,
            FN=fn,
            Confused=confused,
            count_TP=len(tp),
            count_FP=len(fp),
            count_FN=len(fn),
            count_Confused=len(confused),
            has_TP=len(tp) > 0,
            has_FP=len(fp) > 0,
            has_FN=len(fn) > 0,
            has_Confused=len(confused) > 0,
            max_confidence_above_t=max(scores) if len(scores) > 0 else None,
            min_confidence_above_t=min(scores) if len(scores) > 0 else None,
            **{threshold_key(label): value for label, value in thresholds.items()},
        )

    def compute_f1_optimal_thresholds(
        self,
        configuration: ThresholdConfiguration,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
    ) -> None:
        if configuration.threshold_strategy != ThresholdStrategy.F1_OPTIMAL:
            return

        if configuration.display_name() in self.threshold_cache.keys():
            return

        all_bbox_matches = [
            match_inferences_multiclass(
                ground_truth.bboxes,
                [inf for inf in inference.bboxes if inf.score >= configuration.min_confidence_score],
                ignored_ground_truths=ground_truth.ignored_bboxes,
                mode="pascal",
                iou_threshold=configuration.iou_threshold,
            )
            for _, ground_truth, inference in inferences
        ]
        optimal_thresholds = compute_optimal_f1(all_bbox_matches)
        self.threshold_cache[configuration.display_name()] = optimal_thresholds

    def compute_test_sample_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        configuration: Optional[ThresholdConfiguration] = None,
    ) -> List[Tuple[TestSample, TestSampleMetrics]]:
        assert configuration is not None, "must specify configuration"
        # compute thresholds to cache values for subsequent steps
        self.compute_f1_optimal_thresholds(configuration=configuration, inferences=inferences)
        labels = {gt.label for _, gts, _ in inferences for gt in gts.bboxes}
        return [(ts, self.compute_image_metrics(gt, inf, configuration, labels)) for ts, gt, inf in inferences]

    def compute_aggregate_label_metrics(
        self,
        matchings: List[MulticlassInferenceMatches],
        thresholds: Dict[str, float],
        label: str,
    ) -> ClassMetricsPerTestCase:
        m_matched = []
        m_unmatched_gt = []
        m_unmatched_inf = []
        tpr_counter = 0
        fpr_counter = 0
        confused_count = 0
        matched_fns = 0
        # filter the matching to only consider one class
        for match in matchings:
            has_tp = False
            has_fp = False
            for gt, inf in match.matched:
                if gt.label == label and inf.score >= thresholds[inf.label]:
                    m_matched.append((gt, inf))
                    has_tp = True
                elif gt.label == label:
                    matched_fns += 1
            for gt, inf in match.unmatched_gt:
                if gt.label == label:
                    m_unmatched_gt.append((gt, inf))
                    if inf is not None and inf.score >= thresholds[inf.label]:
                        confused_count += 1
            for inf in match.unmatched_inf:
                if inf.label == label and inf.score >= thresholds[inf.label]:
                    m_unmatched_inf.append(inf)
                    has_fp = True
            if has_tp:
                tpr_counter += 1
            if has_fp:
                fpr_counter += 1
        tp_count = len(m_matched)
        fp_count = len(m_unmatched_inf)
        fn_count = len(m_unmatched_gt) + matched_fns
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        all_bbox_matches = [
            MulticlassInferenceMatches(matched=m_matched, unmatched_gt=m_unmatched_gt, unmatched_inf=m_unmatched_inf),
        ]
        baseline_pr_plot = compute_pr_plot(all_bbox_matches)
        baseline_pr_plot = compute_pr_plot(all_bbox_matches).curves[0] if baseline_pr_plot is not None else None

        average_precision = compute_ap(baseline_pr_plot.y, baseline_pr_plot.x) if baseline_pr_plot is not None else 0

        return ClassMetricsPerTestCase(
            Class=label,
            Objects=tp_count + fn_count,
            Inferences=tp_count + fp_count,
            TP=tp_count,
            FN=fn_count,
            FP=fp_count,
            Confused=confused_count,
            TPR=tpr_counter / len(matchings) if len(matchings) > 0 else 0.0,
            FPR=fpr_counter / len(matchings) if len(matchings) > 0 else 0.0,
            Precision=precision,
            Recall=recall,
            F1=2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0,
            AP=average_precision,
        )

    def compute_test_case_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        metrics: List[TestSampleMetrics],
        configuration: Optional[ThresholdConfiguration] = None,
    ) -> TestCaseMetrics:
        assert configuration is not None, "must specify configuration"
        thresholds = self.get_confidence_thresholds(configuration)
        self.locators_by_test_case[test_case.name] = [ts.locator for ts, _, _ in inferences]
        tp_count = sum(im.count_TP for im in metrics)
        fp_count = sum(im.count_FP for im in metrics)
        fn_count = sum(im.count_FN for im in metrics)
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0

        all_bbox_matches = [
            match_inferences_multiclass(
                ground_truth.bboxes,
                [inf for inf in inference.bboxes if inf.score >= configuration.min_confidence_score],
                ignored_ground_truths=ground_truth.ignored_bboxes,
                mode="pascal",
                iou_threshold=configuration.iou_threshold,
            )
            for _, ground_truth, inference in inferences
        ]

        baseline_pr_plot = compute_pr_plot(all_bbox_matches)
        baseline_pr_plot = compute_pr_plot(all_bbox_matches).curves[0] if baseline_pr_plot is not None else None
        average_precision = compute_ap(baseline_pr_plot.y, baseline_pr_plot.x) if baseline_pr_plot is not None else 0

        # compute nested metrics per class
        labels = {gt.label for _, gts, _ in inferences for gt in gts.bboxes}
        per_class_metrics: List[ClassMetricsPerTestCase] = []
        for label in labels:
            metrics_per_class = self.compute_aggregate_label_metrics(all_bbox_matches, thresholds, label)
            per_class_metrics.append(metrics_per_class)

        return TestCaseMetrics(
            PerClass=per_class_metrics,
            Objects=tp_count + fn_count,
            Inferences=tp_count + fp_count,
            TP=tp_count,
            FN=fn_count,
            FP=fp_count,
            TPR=sum(1 for im in metrics if im.has_TP) / len(metrics) if len(metrics) > 0 else 0.0,
            FPR=sum(1 for im in metrics if im.has_FP) / len(metrics) if len(metrics) > 0 else 0.0,
            Precision=precision,
            Recall=recall,
            F1=2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0,
            AP=average_precision,
        )

    def compute_test_case_plots(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        metrics: List[TestSampleMetrics],
        configuration: Optional[ThresholdConfiguration] = None,
    ) -> Optional[List[Plot]]:
        assert configuration is not None, "must specify configuration"
        all_bbox_matches = [
            match_inferences_multiclass(
                ground_truth.bboxes,
                [inf for inf in inference.bboxes if inf.score >= configuration.min_confidence_score],
                ignored_ground_truths=ground_truth.ignored_bboxes,
                mode="pascal",
                iou_threshold=configuration.iou_threshold,
            )
            for _, ground_truth, inference in inferences
        ]

        return [
            plot
            for plot in [*compute_pr_f1_plots(all_bbox_matches), compute_confusion_matrix_plot(all_bbox_matches)]
            if plot is not None
        ]

    def compute_test_suite_metrics(
        self,
        test_suite: TestSuite,
        metrics: List[Tuple[TestCase, TestCaseMetrics]],
        configuration: Optional[ThresholdConfiguration] = None,
    ) -> TestSuiteMetrics:
        assert configuration is not None, "must specify configuration"
        return TestSuiteMetrics(
            n_images=len({locator for tc, _ in metrics for locator in self.locators_by_test_case[tc.name]}),
            mean_AP=np.average([tcm.AP for _, tcm in metrics]),
            variance_AP=np.var([tcm.AP for _, tcm in metrics]),
        )

    def get_confidence_thresholds(self, configuration: ThresholdConfiguration) -> Dict[str, float]:
        # if configuration.threshold_strategy == ThresholdStrategy.FIXED_05:
        #     return defaultdict(lambda: 0.5)
        # if configuration.threshold_strategy == ThresholdStrategy.FIXED_075:
        #     return defaultdict(lambda: 0.75)
        if configuration.threshold_strategy == ThresholdStrategy.F1_OPTIMAL:
            return self.threshold_cache[configuration.display_name()]
        raise RuntimeError(f"unrecognized threshold strategy: {configuration.threshold_strategy}")
