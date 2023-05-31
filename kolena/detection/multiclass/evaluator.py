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

from kolena.detection.multiclass._utils import threshold_key
from kolena.detection.multiclass.workflow import GroundTruth
from kolena.detection.multiclass.workflow import Inference
from kolena.detection.multiclass.workflow import TestCase
from kolena.detection.multiclass.workflow import TestCaseMetrics
from kolena.detection.multiclass.workflow import TestSample
from kolena.detection.multiclass.workflow import TestSampleMetrics
from kolena.detection.multiclass.workflow import ThresholdConfiguration
from kolena.detection.multiclass.workflow import ThresholdStrategy
from kolena.workflow import EvaluationResults
from kolena.workflow import Plot
from kolena.workflow import TestCases
from kolena.workflow.metrics import match_inferences_multiclass

# from kolena.detection.multiclass.workflow import ClassMetricsPerTestCase
# from kolena.detection.multiclass.workflow import TestSuiteMetrics
# from kolena.workflow import ConfusionMatrix
# from kolena.workflow import Curve
# from kolena.workflow import CurvePlot

Result = Tuple[TestSample, GroundTruth, Inference]

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


def compute_image_metrics(
    configuration: ThresholdConfiguration,
    ground_truth: GroundTruth,
    inference: Inference,
) -> TestSampleMetrics:
    thresholds = get_confidence_thresholds(configuration)
    bbox_matches = match_inferences_multiclass(ground_truth.bboxes, inference.bboxes, configuration)
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
        has_TN=len(inference.bboxes) == 0 and len(ground_truth.bboxes) == 0,
        has_Confused=len(confused) > 0,
        max_confidence_above_t=max(scores) if len(scores) > 0 else None,
        min_confidence_above_t=min(scores) if len(scores) > 0 else None,
        **{threshold_key(label): value for label, value in thresholds.items()},
    )


def compute_test_sample_metrics(
    inferences: List[Tuple[TestSample, GroundTruth, Inference]],
    configuration: ThresholdConfiguration,
) -> List[Tuple[TestSample, TestSampleMetrics]]:
    assert configuration is not None, "must specify configuration"
    # compute thresholds to cache values for subsequent steps
    # compute_f1_optimal_thresholds(configuration=configuration, inferences=inferences)
    labels = {gt.label for _, gts, _ in inferences for gt in gts.bboxes}
    return [(ts, compute_image_metrics(gt, inf, configuration, labels)) for ts, gt, inf in inferences]


def MulticlassDetectionEvaluator(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
    configuration: ThresholdConfiguration = dataclasses.field(default_factory=ThresholdConfiguration),
) -> EvaluationResults:
    labels_set: Set[str] = set()
    for gt in ground_truths:
        labels_set.add(gt.label)
    for inf in inferences:
        for label in inf.inferences:
            labels_set.add(label.label)
    labels = sorted(labels_set)
    print(labels)

    metrics_test_sample: List[Tuple[TestSample, TestSampleMetrics]] = [
        (ts, compute_test_sample_metrics(configuration, gt, inf))
        for ts, gt, inf in zip(test_samples, ground_truths, inferences)
    ]
    test_sample_metrics: List[TestSampleMetrics] = [mts for _, mts in metrics_test_sample]
    print(test_sample_metrics)
    metrics_test_case: List[Tuple[TestCase, TestCaseMetrics]] = []
    plots_test_case: List[Tuple[TestCase, List[Plot]]] = []
    # for tc, tc_samples, tc_gts, tc_infs, tc_metrics in test_cases.iter(
    #     test_samples,
    #     ground_truths,
    #     inferences,
    #     test_sample_metrics,
    # ):
    #     aggregated_label_metrics = _aggregate_label_metrics(labels, tc_samples, tc_gts, tc_infs, tc_metrics)
    #     test_case_metrics = _compute_test_case_metrics(tc_samples, tc_gts, tc_metrics, aggregated_label_metrics)
    #     metrics_test_case.append((tc, test_case_metrics))
    #     test_case_plots = _compute_test_case_plots(
    #         tc.name,
    #         labels,
    #         tc_gts,
    #         tc_infs,
    #         tc_metrics,
    #         aggregated_label_metrics,
    #         confidence_range,
    #     )
    #     plots_test_case.append((tc, test_case_plots))

    # all_test_case_metrics = [metric for _, metric in metrics_test_case]
    # metrics_test_suite = _compute_test_suite_metrics(
    #     labels,
    #     test_samples,
    #     ground_truths,
    #     inferences,
    #     test_sample_metrics,
    #     all_test_case_metrics,
    # )

    return EvaluationResults(
        metrics_test_sample=metrics_test_sample,
        metrics_test_case=metrics_test_case,
        plots_test_case=plots_test_case,
        metrics_test_suite=None,
    )
