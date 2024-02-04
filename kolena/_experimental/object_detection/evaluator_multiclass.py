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

import numpy as np

from kolena._experimental.object_detection import ClassMetricsPerTestCase
from kolena._experimental.object_detection import GroundTruth
from kolena._experimental.object_detection import Inference
from kolena._experimental.object_detection import TestCase
from kolena._experimental.object_detection import TestCaseMetrics
from kolena._experimental.object_detection import TestSample
from kolena._experimental.object_detection import TestSampleMetrics
from kolena._experimental.object_detection import TestSuite
from kolena._experimental.object_detection import TestSuiteMetrics
from kolena._experimental.object_detection import ThresholdConfiguration
from kolena._experimental.object_detection.utils import compute_average_precision
from kolena._experimental.object_detection.utils import compute_confusion_matrix_plot
from kolena._experimental.object_detection.utils import compute_f1_plot_multiclass
from kolena._experimental.object_detection.utils import compute_optimal_f1_threshold_multiclass
from kolena._experimental.object_detection.utils import compute_pr_curve
from kolena._experimental.object_detection.utils import compute_pr_plot_multiclass
from kolena._experimental.object_detection.utils import filter_inferences
from kolena.workflow import Evaluator
from kolena.workflow import EvaluatorConfiguration
from kolena.workflow import Plot
from kolena.workflow.annotation import ScoredLabel
from kolena.workflow.metrics import f1_score as compute_f1_score
from kolena.workflow.metrics import match_inferences_multiclass
from kolena.workflow.metrics import MulticlassInferenceMatches
from kolena.workflow.metrics import precision as compute_precision
from kolena.workflow.metrics import recall as compute_recall


class MulticlassObjectDetectionEvaluator(Evaluator):
    """
    The `MulticlassObjectDetectionEvaluator` transforms inferences into metrics for the object detection workflow for
    multiple classes.

    When a [`ThresholdConfiguration`][kolena._experimental.object_detection.workflow.ThresholdConfiguration] is
    configured to use an F1-Optimal threshold strategy, the evaluator requires that the first test case retrieved for
    a test suite contains the complete sample set.

    For additional functionality, see the associated [base class documentation][kolena.workflow.evaluator.Evaluator].
    """

    threshold_cache: Dict[str, Dict[str, float]]  # configuration -> label -> threshold
    """
    Assumes that the first test case retrieved for the test suite contains the complete sample set to be used for
    F1-Optimal threshold computation. Subsequent requests for a given threshold strategy (for other test cases) will
    hit this cache and use the previously computed population level confidence thresholds.
    """

    locators_by_test_case: Dict[str, List[str]]
    """
    Keeps track of test sample locators for each test case (used for total # of image count in aggregated metrics).
    """

    matchings_by_test_case: Dict[str, Dict[str, List[MulticlassInferenceMatches]]]
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
    ) -> TestSampleMetrics:
        return TestSampleMetrics(
            TP=[],
            FP=[],
            FN=[],
            Confused=[],
            count_TP=0,
            count_FP=0,
            count_FN=0,
            count_Confused=0,
            has_TP=False,
            has_FP=False,
            has_FN=False,
            has_Confused=False,
            ignored=True,
            max_confidence_above_t=None,
            min_confidence_above_t=None,
            thresholds=[],
        )

    def test_sample_metrics(
        self,
        bbox_matches: MulticlassInferenceMatches,
        thresholds: Dict[str, float],
    ) -> TestSampleMetrics:
        tp = [inf for _, inf in bbox_matches.matched if inf.score >= thresholds[inf.label]]
        fp = [inf for inf in bbox_matches.unmatched_inf if inf.score >= thresholds[inf.label]]
        fn = [gt for gt, _ in bbox_matches.unmatched_gt] + [
            gt for gt, inf in bbox_matches.matched if inf.score < thresholds[inf.label]
        ]
        confused = [
            inf for _, inf in bbox_matches.unmatched_gt if inf is not None and inf.score >= thresholds[inf.label]
        ]
        non_ignored_inferences = tp + fp
        scores = [inf.score for inf in non_ignored_inferences]
        inference_labels = {inf.label for _, inf in bbox_matches.matched} | {
            inf.label for inf in bbox_matches.unmatched_inf
        }
        fields = [
            ScoredLabel(label=label, score=thresholds[label])
            for label in sorted(thresholds.keys())
            if label in inference_labels
        ]
        return TestSampleMetrics(
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
            ignored=False,
            max_confidence_above_t=max(scores) if len(scores) > 0 else None,
            min_confidence_above_t=min(scores) if len(scores) > 0 else None,
            thresholds=fields,
        )

    def compute_image_metrics(
        self,
        ground_truth: GroundTruth,
        inference: Inference,
        configuration: ThresholdConfiguration,
        test_case_name: str,
    ) -> TestSampleMetrics:
        assert configuration is not None, "must specify configuration"
        thresholds = self.get_confidence_thresholds(configuration)
        if inference.ignored:
            return self.test_sample_metrics_ignored()

        filtered_inferences = filter_inferences(
            inferences=inference.bboxes,
            confidence_score=configuration.min_confidence_score,
        )
        bbox_matches: MulticlassInferenceMatches = match_inferences_multiclass(
            ground_truth.bboxes,
            filtered_inferences,
            ignored_ground_truths=ground_truth.ignored_bboxes,
            mode="pascal",
            iou_threshold=configuration.iou_threshold,
        )
        self.matchings_by_test_case[configuration.display_name()][test_case_name].append(bbox_matches)

        return self.test_sample_metrics(bbox_matches, thresholds)

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
            match_inferences_multiclass(
                ground_truth.bboxes,
                filter_inferences(inferences=inference.bboxes, confidence_score=configuration.min_confidence_score),
                ignored_ground_truths=ground_truth.ignored_bboxes,
                mode="pascal",
                iou_threshold=configuration.iou_threshold,
            )
            for _, ground_truth, inference in inferences
            if not inference.ignored
        ]
        optimal_thresholds = compute_optimal_f1_threshold_multiclass(all_bbox_matches)
        self.threshold_cache[configuration.display_name()] = defaultdict(
            lambda: configuration.min_confidence_score,
            optimal_thresholds,
        )

    def compute_test_sample_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],  # type: ignore
        configuration: Optional[ThresholdConfiguration] = None,  # type: ignore
    ) -> List[Tuple[TestSample, TestSampleMetrics]]:
        assert configuration is not None, "must specify configuration"
        # compute thresholds to cache values for subsequent steps
        self.compute_and_cache_f1_optimal_thresholds(configuration, inferences)
        return [
            (ts, self.compute_image_metrics(gt, inf, configuration, test_case.name))  # type: ignore
            for ts, gt, inf in inferences
        ]

    def bbox_matches_and_count_for_one_label(
        self,
        matchings: List[MulticlassInferenceMatches],
        label: str,
    ) -> Tuple[MulticlassInferenceMatches, int]:
        match_matched = []
        match_unmatched_gt = []
        match_unmatched_inf = []
        samples_count = 0
        # filter the matching to only consider one class
        for match in matchings:
            sample_flag = False
            for gt, inf in match.matched:
                if gt.label == label:
                    sample_flag = True
                    match_matched.append((gt, inf))
            for gt, inf in match.unmatched_gt:
                if gt.label == label:
                    sample_flag = True
                    match_unmatched_gt.append((gt, inf))
            for inf in match.unmatched_inf:
                if inf.label == label:
                    match_unmatched_inf.append(inf)
            if sample_flag:
                samples_count += 1

        bbox_matches_for_one_label = MulticlassInferenceMatches(
            matched=match_matched,
            unmatched_gt=match_unmatched_gt,
            unmatched_inf=match_unmatched_inf,
        )

        return bbox_matches_for_one_label, samples_count

    def class_metrics_per_test_case(
        self,
        label: str,
        thresholds: Dict[str, float],
        class_matches: MulticlassInferenceMatches,
        samples_count: int,
        average_precision: float,
    ) -> ClassMetricsPerTestCase:
        matched = class_matches.matched
        unmatched_gt = class_matches.unmatched_gt
        unmatched_inf = class_matches.unmatched_inf

        tp = [inf for _, inf in matched if inf.score >= thresholds[inf.label]]
        fp = [inf for inf in unmatched_inf if inf.score >= thresholds[inf.label]]
        fn = [gt for gt, _ in unmatched_gt] + [gt for gt, inf in matched if inf.score < thresholds[inf.label]]
        tp_count = len(tp)
        fp_count = len(fp)
        fn_count = len(fn)
        precision = compute_precision(tp_count, fp_count)
        recall = compute_recall(tp_count, fn_count)
        f1_score = compute_f1_score(tp_count, fp_count, fn_count)

        return ClassMetricsPerTestCase(
            Class=label,
            nImages=samples_count,
            Threshold=thresholds[label],
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

    def compute_aggregate_label_metrics(
        self,
        matchings: List[MulticlassInferenceMatches],
        label: str,
        thresholds: Dict[str, float],
    ) -> ClassMetricsPerTestCase:
        class_matches, samples_count = self.bbox_matches_and_count_for_one_label(matchings, label)

        average_precision = 0.0
        baseline_pr_curve = compute_pr_curve([class_matches])
        if baseline_pr_curve is not None:
            average_precision = compute_average_precision(baseline_pr_curve.y, baseline_pr_curve.x)  # type: ignore

        return self.class_metrics_per_test_case(label, thresholds, class_matches, samples_count, average_precision)

    def test_case_metrics(
        self,
        per_class_metrics: List[ClassMetricsPerTestCase],
        metrics: List[TestSampleMetrics],
    ) -> TestCaseMetrics:
        tp_count = sum(im.count_TP for im in metrics)
        fp_count = sum(im.count_FP for im in metrics)
        fn_count = sum(im.count_FN for im in metrics)
        ignored_count = sum(1 if im.ignored else 0 for im in metrics)
        macro_prec_data = np.mean([data.Precision for data in per_class_metrics])
        macro_rec_data = np.mean([data.Recall for data in per_class_metrics])
        return TestCaseMetrics(
            PerClass=per_class_metrics,
            Objects=tp_count + fn_count,
            Inferences=tp_count + fp_count,
            TP=tp_count,
            FN=fn_count,
            FP=fp_count,
            nIgnored=ignored_count,
            macro_Precision=macro_prec_data if per_class_metrics else 0.0,  # type: ignore
            macro_Recall=macro_rec_data if per_class_metrics else 0.0,  # type: ignore
            macro_F1=np.mean([data.F1 for data in per_class_metrics]) if per_class_metrics else 0.0,  # type: ignore
            mean_AP=np.mean([data.AP for data in per_class_metrics]) if per_class_metrics else 0.0,  # type: ignore
            micro_Precision=compute_precision(tp_count, fp_count),
            micro_Recall=compute_recall(tp_count, fn_count),
            micro_F1=compute_f1_score(tp_count, fp_count, fn_count),
        )

    def compute_test_case_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],  # type: ignore
        metrics: List[TestSampleMetrics],  # type: ignore
        configuration: Optional[ThresholdConfiguration] = None,  # type: ignore
    ) -> TestCaseMetrics:
        assert configuration is not None, "must specify configuration"
        thresholds = self.get_confidence_thresholds(configuration)
        all_bbox_matches = self.matchings_by_test_case[configuration.display_name()][test_case.name]  # type: ignore
        self.locators_by_test_case[test_case.name] = [ts.locator for ts, _, _ in inferences]  # type: ignore

        # compute nested metrics per class
        labels = {gt.label for _, gts, _ in inferences for gt in gts.bboxes} | {
            inf.label for _, _, infs in inferences for inf in infs.bboxes
        }
        per_class_metrics: List[ClassMetricsPerTestCase] = []
        for label in sorted(labels):
            metrics_per_class = self.compute_aggregate_label_metrics(all_bbox_matches, label, thresholds)
            per_class_metrics.append(metrics_per_class)

        return self.test_case_metrics(per_class_metrics, metrics)

    def compute_test_case_plots(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],  # type: ignore
        metrics: List[TestSampleMetrics],  # type: ignore
        configuration: Optional[ThresholdConfiguration] = None,  # type: ignore
    ) -> Optional[List[Plot]]:
        assert configuration is not None, "must specify configuration"
        thresholds = self.get_confidence_thresholds(configuration)
        all_bbox_matches = self.matchings_by_test_case[configuration.display_name()][test_case.name]  # type: ignore

        # clean matching for confusion matrix
        match_matched = [
            (gt, inf) for match in all_bbox_matches for gt, inf in match.matched if inf.score >= thresholds[inf.label]
        ]
        match_unmatched_gt = [
            (gt, inf)
            for match in all_bbox_matches
            for gt, inf in match.unmatched_gt
            if inf is not None and inf.score >= thresholds[inf.label]
        ]
        confusion_matrix_matchings = [
            MulticlassInferenceMatches(
                matched=match_matched,
                unmatched_gt=match_unmatched_gt,
                unmatched_inf=[],
            ),
        ]

        plots: List[Plot] = []
        plots.extend(
            filter(
                None,
                [
                    compute_f1_plot_multiclass(all_bbox_matches),
                    compute_pr_plot_multiclass(all_bbox_matches),
                    compute_confusion_matrix_plot(confusion_matrix_matchings),
                ],
            ),
        )

        return plots

    def test_suite_metrics(self, unique_locators: Set[str], average_precisions: List[float]) -> TestSuiteMetrics:
        return TestSuiteMetrics(
            n_images=len(unique_locators),
            mean_AP=np.mean(average_precisions) if average_precisions else 0.0,  # type: ignore
        )

    def compute_test_suite_metrics(
        self,
        test_suite: TestSuite,
        metrics: List[Tuple[TestCase, TestCaseMetrics]],  # type: ignore
        configuration: Optional[ThresholdConfiguration] = None,  # type: ignore
    ) -> TestSuiteMetrics:
        assert configuration is not None, "must specify configuration"
        unique_locators = {
            locator for tc, _ in metrics for locator in self.locators_by_test_case[tc.name]  # type: ignore
        }
        average_precisions = [tcm.mean_AP for _, tcm in metrics]
        return self.test_suite_metrics(unique_locators, average_precisions)

    def get_confidence_thresholds(self, configuration: ThresholdConfiguration) -> Dict[str, float]:
        if configuration.threshold_strategy == "F1-Optimal":
            return self.threshold_cache[configuration.display_name()]
        else:
            return defaultdict(lambda: configuration.threshold_strategy)  # type: ignore
