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
from dataclasses import make_dataclass
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import numpy as np

from kolena._experimental.object_detection.utils import clean_threshold_key
from kolena._experimental.object_detection.utils import compute_average_precision
from kolena._experimental.object_detection.utils import compute_confusion_matrix_plot
from kolena._experimental.object_detection.utils import compute_f1_plot
from kolena._experimental.object_detection.utils import compute_f1_plot_multiclass
from kolena._experimental.object_detection.utils import compute_optimal_f1_threshold_multiclass
from kolena._experimental.object_detection.utils import compute_pr_curve
from kolena._experimental.object_detection.utils import compute_pr_plot
from kolena._experimental.object_detection.utils import compute_pr_plot_multiclass
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
from kolena.workflow.metrics import f1_score as compute_f1_score
from kolena.workflow.metrics import InferenceMatches
from kolena.workflow.metrics import match_inferences_multiclass
from kolena.workflow.metrics import MulticlassInferenceMatches
from kolena.workflow.metrics import precision as compute_precision
from kolena.workflow.metrics import recall as compute_recall


class ObjectDetectionEvaluator(Evaluator):
    """
    This `ObjectDetectionEvaluator` transforms inferences into metrics for the object detection workflow for a single
    class or multiple classes.

    When a [`ThresholdConfiguration`][kolena._experimental.object_detection.workflow.ThresholdConfiguration] is
    configured to use an F1-Optimal threshold strategy, the evaluator requires that the first test case retrieved for
    a test suite contains the complete sample set.

    For additional functionality, see the associated
    [base class documentation][kolena.workflow.evaluator.Evaluator].
    """

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
        assert configuration is not None, "must specify configuration"
        thresholds = self.get_confidence_thresholds(configuration)
        bbox_matches: MulticlassInferenceMatches = match_inferences_multiclass(
            ground_truth.bboxes,
            [
                inf
                for inf in inference.bboxes
                if inf.label in labels and inf.score >= max(configuration.min_confidence_score, thresholds[inf.label])
            ],
            ignored_ground_truths=ground_truth.ignored_bboxes,
            mode="pascal",
            iou_threshold=configuration.iou_threshold,
        )
        tp = [inf for _, inf in bbox_matches.matched]
        fp = bbox_matches.unmatched_inf
        fn = [gt for gt, _ in bbox_matches.unmatched_gt]

        confused = [inf for _, inf in bbox_matches.unmatched_gt if inf is not None]
        non_ignored_inferences = tp + fp
        scores = [inf.score for inf in non_ignored_inferences]
        image_labels = {gt.label for gt in ground_truth.bboxes}
        fields = [(clean_threshold_key(label), float) for label in thresholds.keys() if label in image_labels]
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
            **{clean_threshold_key(label): value for label, value in thresholds.items() if label in image_labels},
        )

    def compute_and_cache_f1_optimal_thresholds(
        self,
        configuration: ThresholdConfiguration,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
    ) -> None:
        if configuration.threshold_strategy != ThresholdStrategy.F1_OPTIMAL:
            return

        if configuration.display_name() in self.threshold_cache.keys():
            return
        labels = {gt.label for _, gts, _ in inferences for gt in gts.bboxes}
        all_bbox_matches = [
            match_inferences_multiclass(
                ground_truth.bboxes,
                [
                    inf
                    for inf in inference.bboxes
                    if inf.label in labels and inf.score >= configuration.min_confidence_score
                ],
                ignored_ground_truths=ground_truth.ignored_bboxes,
                mode="pascal",
                iou_threshold=configuration.iou_threshold,
            )
            for _, ground_truth, inference in inferences
        ]
        optimal_thresholds = compute_optimal_f1_threshold_multiclass(all_bbox_matches)
        self.threshold_cache[configuration.display_name()] = optimal_thresholds

    def compute_test_sample_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        configuration: Optional[ThresholdConfiguration] = None,
    ) -> List[Tuple[TestSample, TestSampleMetrics]]:
        assert configuration is not None, "must specify configuration"
        # compute thresholds to cache values for subsequent steps
        self.compute_and_cache_f1_optimal_thresholds(configuration, inferences)
        labels = {gt.label for _, gts, _ in inferences for gt in gts.bboxes}
        return [(ts, self.compute_image_metrics(gt, inf, configuration, labels)) for ts, gt, inf in inferences]

    def compute_aggregate_label_metrics(
        self,
        matchings: List[MulticlassInferenceMatches],
        label: str,
        thresholds: Dict[str, float],
    ) -> ClassMetricsPerTestCase:
        inf_match_matched = []
        inf_match_unmatched_gt = []
        inf_match_unmatched_inf = []
        tpr_counter = 0
        fpr_counter = 0
        confused_counter = 0
        samples = 0
        # filter the matching to only consider one class
        for match in matchings:
            has_tp = False
            has_fn = False
            has_fp = False
            for gt, inf in match.matched:
                if gt.label == label:
                    inf_match_matched.append((gt, inf))
                    tpr_counter += 1
            for gt, inf in match.unmatched_gt:
                if gt.label == label:
                    inf_match_unmatched_gt.append(gt)
                    has_fn = True
                    if inf is not None:
                        confused_counter += 1
            for inf in match.unmatched_inf:
                if inf.label == label:
                    inf_match_unmatched_inf.append(inf)
                    has_fp = True
            if has_tp:
                tpr_counter += 1
            if has_fp:
                fpr_counter += 1
            if has_tp or has_fn or has_fp:
                samples += 1
        tp_count = len(inf_match_matched)
        fp_count = len(inf_match_unmatched_inf)
        fn_count = len(inf_match_unmatched_gt)
        precision = compute_precision(tp_count, fp_count)
        recall = compute_recall(tp_count, fn_count)
        f1_score = compute_f1_score(tp_count, fp_count, fn_count)
        all_bbox_matches = [
            InferenceMatches(
                matched=inf_match_matched,
                unmatched_gt=inf_match_unmatched_gt,
                unmatched_inf=inf_match_unmatched_inf,
            ),
        ]

        average_precision = 0.0
        if precision > 0:
            baseline_pr_curve = compute_pr_curve(all_bbox_matches)
            if baseline_pr_curve is not None:
                average_precision = compute_average_precision(baseline_pr_curve.y, baseline_pr_curve.x)

        return ClassMetricsPerTestCase(
            Class=label,
            Threshold=thresholds[label],
            TestSamples=samples,
            Objects=tp_count + fn_count,
            Inferences=tp_count + fp_count,
            TP=tp_count,
            FN=fn_count,
            FP=fp_count,
            Confused=confused_counter,
            Precision=precision,
            Recall=recall,
            F1=f1_score,
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
        labels = {gt.label for _, gts, _ in inferences for gt in gts.bboxes}
        all_bbox_matches = [
            match_inferences_multiclass(
                ground_truth.bboxes,
                [
                    inf
                    for inf in inference.bboxes
                    if inf.label in labels
                    and inf.score >= max(configuration.min_confidence_score, thresholds[inf.label])
                ],
                ignored_ground_truths=ground_truth.ignored_bboxes,
                mode="pascal",
                iou_threshold=configuration.iou_threshold,
            )
            for _, ground_truth, inference in inferences
        ]
        self.locators_by_test_case[test_case.name] = [ts.locator for ts, _, _ in inferences]
        tp_count = sum(im.count_TP for im in metrics)
        fp_count = sum(im.count_FP for im in metrics)
        fn_count = sum(im.count_FN for im in metrics)
        precision = compute_precision(tp_count, fp_count)
        recall = compute_recall(tp_count, fn_count)
        f1_score = compute_f1_score(tp_count, fp_count, fn_count)

        average_precision = 0.0
        if precision > 0:
            baseline_pr_curve = compute_pr_curve(all_bbox_matches)
            if baseline_pr_curve is not None:
                average_precision = compute_average_precision(baseline_pr_curve.y, baseline_pr_curve.x)

        # compute nested metrics per class
        per_class_metrics: List[ClassMetricsPerTestCase] = []
        for label in labels:
            metrics_per_class = self.compute_aggregate_label_metrics(all_bbox_matches, label, thresholds)
            per_class_metrics.append(metrics_per_class)

        return TestCaseMetrics(
            TestSamples=len(inferences),
            PerClass=per_class_metrics,
            Objects=tp_count + fn_count,
            Inferences=tp_count + fp_count,
            TP=tp_count,
            FN=fn_count,
            FP=fp_count,
            Precision=precision,
            Recall=recall,
            F1=f1_score,
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
        thresholds = self.get_confidence_thresholds(configuration)
        labels = {gt.label for _, gts, _ in inferences for gt in gts.bboxes}
        all_bbox_matches = [
            match_inferences_multiclass(
                ground_truth.bboxes,
                [
                    inf
                    for inf in inference.bboxes
                    if inf.label in labels
                    and inf.score >= max(configuration.min_confidence_score, thresholds[inf.label])
                ],
                ignored_ground_truths=ground_truth.ignored_bboxes,
                mode="pascal",
                iou_threshold=configuration.iou_threshold,
            )
            for _, ground_truth, inference in inferences
        ]

        plots: Optional[List[Plot]] = []
        plots.extend(
            filter(
                None,
                [
                    compute_f1_plot_multiclass(all_bbox_matches),
                    compute_f1_plot(all_bbox_matches),
                    compute_pr_plot_multiclass(all_bbox_matches),
                    compute_pr_plot(all_bbox_matches),
                    compute_confusion_matrix_plot(all_bbox_matches),
                ],
            ),
        )

        return plots

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
        if configuration.threshold_strategy == ThresholdStrategy.FIXED_03:
            return defaultdict(lambda: 0.3)
        if configuration.threshold_strategy == ThresholdStrategy.FIXED_05:
            return defaultdict(lambda: 0.5)
        if configuration.threshold_strategy == ThresholdStrategy.FIXED_075:
            return defaultdict(lambda: 0.75)
        if configuration.threshold_strategy == ThresholdStrategy.F1_OPTIMAL:
            return self.threshold_cache[configuration.display_name()]
        raise RuntimeError(f"unrecognized threshold strategy: {configuration.threshold_strategy}")
