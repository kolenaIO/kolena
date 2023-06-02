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
import dataclasses
from collections import defaultdict
from dataclasses import make_dataclass
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple

import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support

from kolena._experimental.object_detection.utils import _compute_sklearn_arrays
from kolena._experimental.object_detection.utils import compute_ap
from kolena._experimental.object_detection.utils import compute_confusion_matrix_plot
from kolena._experimental.object_detection.utils import compute_pr_f1_plots
from kolena._experimental.object_detection.utils import threshold_key
from kolena._experimental.object_detection.workflow import ClassMetricsPerTestCase
from kolena._experimental.object_detection.workflow import GroundTruth
from kolena._experimental.object_detection.workflow import Inference
from kolena._experimental.object_detection.workflow import TestCase
from kolena._experimental.object_detection.workflow import TestCaseMetrics
from kolena._experimental.object_detection.workflow import TestSample
from kolena._experimental.object_detection.workflow import TestSampleMetrics
from kolena._experimental.object_detection.workflow import TestSuiteMetrics
from kolena._experimental.object_detection.workflow import ThresholdConfiguration
from kolena._experimental.object_detection.workflow import ThresholdStrategy
from kolena.workflow import EvaluationResults
from kolena.workflow import Plot
from kolena.workflow import TestCases
from kolena.workflow.evaluator import Curve
from kolena.workflow.metrics import match_inferences_multiclass
from kolena.workflow.metrics._geometry import MulticlassInferenceMatches


# Assumes that the first test case retrieved for the test suite contains the complete sample set to be used for
# F1-Optimal threshold computation. Subsequent requests for a given threshold strategy (for other test cases) will
# hit this cache and use the previously computed population level confidence thresholds.
threshold_cache: Dict[str, Dict[str, float]] = {}  # configuration -> label -> threshold

# Keeps track of test sample locators for each test case (used for total # of image count in aggregated metrics)
locators_by_test_case: Dict[str, List[str]] = {}


def get_confidence_thresholds(configuration: ThresholdConfiguration) -> Dict[str, float]:
    if configuration.threshold_strategy == ThresholdStrategy.FIXED_05:
        return defaultdict(lambda: 0.5)
    if configuration.threshold_strategy == ThresholdStrategy.FIXED_075:
        return defaultdict(lambda: 0.75)
    if configuration.threshold_strategy == ThresholdStrategy.F1_OPTIMAL:
        return threshold_cache[configuration.display_name()]
    raise RuntimeError(f"unrecognized threshold strategy: {configuration.threshold_strategy}")


def compute_f1_optimal_thresholds(
    all_bbox_matches: List[MulticlassInferenceMatches],
    configuration: ThresholdConfiguration,
) -> None:
    if configuration.threshold_strategy != ThresholdStrategy.F1_OPTIMAL:
        return

    if configuration.display_name() in threshold_cache.keys():
        return

    optimal_thresholds: Dict[str, float] = {}

    _, _, arrays_by_label = _compute_sklearn_arrays(all_bbox_matches)
    for label in sorted(arrays_by_label.keys()):
        y_true, y_score = arrays_by_label[label]
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        if thresholds[0] < 0:
            precision = precision[1:]
            recall = recall[1:]
            thresholds = thresholds[1:]
        if len(thresholds) == 0:
            optimal_thresholds[label] = -1
        f1_scores = 2 * precision * recall / (precision + recall)
        max_f1_index = np.argmax(f1_scores)
        optimal_thresholds[label] = thresholds[max_f1_index]

    threshold_cache[configuration.display_name()] = optimal_thresholds


def compute_image_metrics(
    bbox_matches: MulticlassInferenceMatches,
    thresholds: Dict[str, float],
) -> TestSampleMetrics:
    tp = [inf for _, inf in bbox_matches.matched if inf.score >= thresholds[inf.label]]
    fp = [inf for inf in bbox_matches.unmatched_inf if inf.score >= thresholds[inf.label]]
    fn = [gt for gt, _ in bbox_matches.unmatched_gt] + [
        gt for gt, inf in bbox_matches.matched if inf.score < thresholds[inf.label]
    ]
    confused = [inf for _, inf in bbox_matches.unmatched_gt if inf is not None and inf.score >= thresholds[inf.label]]
    non_ignored_inferences = [inf for _, inf in bbox_matches.matched] + bbox_matches.unmatched_inf
    scores = [inf.score for inf in non_ignored_inferences if inf.score >= thresholds[inf.label]]
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
        match_matched_gt=[gt for gt, _ in bbox_matches.matched],
        match_matched_inf=[inf for _, inf in bbox_matches.matched],
        match_unmatched_inf=[inf for inf in bbox_matches.unmatched_inf],
        match_unmatched_gt=[gt for gt, _ in bbox_matches.unmatched_gt],
        match_confused_inf=[inf for _, inf in bbox_matches.unmatched_gt],
        **{threshold_key(label): value for label, value in thresholds.items()},
    )


def compute_test_sample_metrics(
    results: List[Tuple[TestSample, GroundTruth, Inference]],
    configuration: ThresholdConfiguration,
) -> List[Tuple[TestSample, TestSampleMetrics]]:
    all_bbox_matches = [
        match_inferences_multiclass(
            gt.bboxes,
            inf.bboxes,
            ignored_ground_truths=gt.ignored_bboxes,
            iou_threshold=configuration.iou_threshold,
        )
        for _, gt, inf in results
    ]
    # compute thresholds to cache values for subsequent steps
    compute_f1_optimal_thresholds(all_bbox_matches, configuration)
    thresholds = get_confidence_thresholds(configuration)
    return [
        (ts, compute_image_metrics(bbox_matches, thresholds))
        for ts, _, _, bbox_matches in zip(results, all_bbox_matches)
    ]


def compute_aggregate_label_metrics(
    tc_matchings: List[MulticlassInferenceMatches],
    label: str,
) -> ClassMetricsPerTestCase:
    m_matched = []
    m_unmatched_gt = []
    m_unmatched_inf = []
    tpr_counter = 0
    fpr_counter = 0

    # filter the matching to only consider one class
    for match in tc_matchings:
        has_tp = False
        has_fp = False
        for gt, inf in match.matched:
            if gt.label == label:
                m_matched.append((gt, inf))
                has_tp = True
        for gt, inf in match.unmatched_gt:
            if gt.label == label:
                m_unmatched_gt.append((gt, inf))
        for inf in match.unmatched_inf:
            if inf.label == label:
                m_unmatched_inf.append(inf)
                has_fp = True
        if has_tp:
            tpr_counter += 1
        if has_fp:
            fpr_counter += 1

    tp_count = len(m_matched)
    fp_count = len(m_unmatched_inf)
    fn_count = len(m_unmatched_gt)
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0

    all_bbox_matches = [
        MulticlassInferenceMatches(matched=m_matched, unmatched_gt=m_unmatched_gt, unmatched_inf=m_unmatched_inf),
    ]
    y_true, y_score, _, _ = _compute_sklearn_arrays(all_bbox_matches)
    thresholds = list(np.linspace(min(abs(y_score)), max(y_score), min(401, len(y_score))))[:-1]
    precisions: List[float] = []
    recalls: List[float] = []
    for threshold in thresholds:
        y_pred = [1 if score > threshold else 0 for score in y_score]
        precision, recall, _, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
        precisions.append(precision)
        recalls.append(recall)
    average_precision = compute_ap(precisions, recalls) if len(precisions) > 0 and len(recalls) > 0 else -1

    return ClassMetricsPerTestCase(
        Class=label,
        Objects=tp_count + fn_count,
        Inferences=tp_count + fp_count,
        TP=tp_count,
        FN=fn_count,
        FP=fp_count,
        TPR=tpr_counter / len(tc_matchings) if len(tc_matchings) > 0 else 0.0,
        FPR=fpr_counter / len(tc_matchings) if len(tc_matchings) > 0 else 0.0,
        Precision=precision,
        Recall=recall,
        F1=2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0,
        AP=average_precision,
    )


def compute_test_case_metrics_and_plots(
    tc_metrics: List[TestSampleMetrics],
    labels: List[str],
) -> Tuple[List[TestCaseMetrics], List[Plot]]:
    tc_matchings = [
        MulticlassInferenceMatches(
            matched=[(gt, inf) for gt, inf in zip(tsm.match_matched_gt, tsm.match_matched_inf)],
            unmatched_gt=[(gt, inf) for gt, inf in zip(tsm.match_unmatched_gt, tsm.match_confused_inf)],
            unmatched_inf=tsm.match_unmatched_inf,
        )
        for tsm in tc_metrics
    ]
    per_class_metrics: List[ClassMetricsPerTestCase] = []
    plots: List[Plot] = compute_pr_f1_plots(tc_matchings, "baseline")
    baseline_pr_plot: Curve = next((curve for curve in plots[1].curves if curve.label == "baseline"), None)
    confusion_matrix = compute_confusion_matrix_plot(tc_matchings)
    if confusion_matrix is not None:
        plots.append(confusion_matrix)

    # compute nested metrics per class
    for label in labels:
        metrics_per_class = compute_aggregate_label_metrics(tc_matchings, label)
        per_class_metrics.append(metrics_per_class)

    tp_count = sum(im.count_TP for im in tc_metrics)
    fp_count = sum(im.count_FP for im in tc_metrics)
    fn_count = sum(im.count_FN for im in tc_metrics)

    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    average_precision = compute_ap(baseline_pr_plot.y, baseline_pr_plot.x)

    return (
        TestCaseMetrics(
            # PerClass=per_class_metrics,
            Objects=tp_count + fn_count,
            Inferences=tp_count + fp_count,
            TP=tp_count,
            FN=fn_count,
            FP=fp_count,
            TPR=sum(1 for im in tc_metrics if im.has_TP) / len(tc_metrics) if len(tc_metrics) > 0 else 0.0,
            FPR=sum(1 for im in tc_metrics if im.has_FP) / len(tc_metrics) if len(tc_metrics) > 0 else 0.0,
            Precision=precision,
            Recall=recall,
            F1=2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0,
            AP=average_precision,
        ),
        plots,
    )


def compute_aggregate_metrics(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
    test_sample_metrics: List[TestSampleMetrics],
    labels: List[str],
) -> Tuple[List[Tuple[TestCase, TestCaseMetrics]], List[Tuple[TestCase, List[Plot]]]]:
    metrics_test_case: List[Tuple[TestCase, TestCaseMetrics]] = []
    plots_test_case: List[Tuple[TestCase, List[Plot]]] = []

    for tc, tc_samples, tc_gts, tc_infs, tc_metrics in test_cases.iter(
        test_samples,
        ground_truths,
        inferences,
        test_sample_metrics,
    ):
        locators_by_test_case[tc.name] = [ts.locator for ts in tc_samples]
        test_case_metrics, test_case_plots = compute_test_case_metrics_and_plots(
            tc_metrics,
            labels,
        )
        metrics_test_case.append((tc, test_case_metrics))
        plots_test_case.append((tc, test_case_plots))

    return metrics_test_case, plots_test_case


def compute_test_suite_metrics(all_test_case_metrics: List[Tuple[TestCase, TestCaseMetrics]]) -> TestSuiteMetrics:
    return TestSuiteMetrics(
        n_images=len({locator for tc, _ in all_test_case_metrics for locator in locators_by_test_case[tc.name]}),
        mean_AP=np.average([tcm.AP for _, tcm in all_test_case_metrics]),
        variance_AP=np.var([tcm.AP for _, tcm in all_test_case_metrics]),
    )


def MulticlassDetectionEvaluator(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
    configuration: ThresholdConfiguration = dataclasses.field(default_factory=ThresholdConfiguration),
) -> EvaluationResults:
    labels_set: Set[str] = set()
    for gt in ground_truths:
        for box in gt.bboxes:
            labels_set.add(box.label)
    for inf in inferences:
        for box in inf.bboxes:
            labels_set.add(box.label)
    labels = sorted(labels_set)

    results = list(zip(test_samples, ground_truths, inferences))
    metrics_test_sample = compute_test_sample_metrics(results, configuration)

    test_sample_metrics: List[TestSampleMetrics] = [mts for _, mts, _ in metrics_test_sample]

    metrics_test_case, plots_test_case = compute_aggregate_metrics(
        test_samples,
        ground_truths,
        inferences,
        test_cases,
        test_sample_metrics,
        configuration,
        labels,
    )

    metrics_test_suite = compute_test_suite_metrics(metrics_test_case)

    return EvaluationResults(
        metrics_test_sample=metrics_test_sample,
        metrics_test_case=metrics_test_case,
        plots_test_case=plots_test_case,
        metrics_test_suite=metrics_test_suite,
    )
