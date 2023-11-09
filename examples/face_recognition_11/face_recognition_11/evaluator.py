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

import numpy as np
from face_recognition_11.utils import calculate_mse_nmse
from face_recognition_11.utils import compute_distances
from face_recognition_11.utils import compute_threshold
from face_recognition_11.utils import create_iou_histogram
from face_recognition_11.utils import create_similarity_histogram
from face_recognition_11.utils import compute_pair_metrics
from face_recognition_11.workflow import GroundTruth
from face_recognition_11.workflow import Inference
from face_recognition_11.workflow import PairSample
from face_recognition_11.workflow import PerBBoxMetrics
from face_recognition_11.workflow import PerKeypointMetrics
from face_recognition_11.workflow import TestCase
from face_recognition_11.workflow import TestCaseMetrics
from face_recognition_11.workflow import TestSample
from face_recognition_11.workflow import TestSampleMetrics
from face_recognition_11.workflow import TestSuiteMetrics
from face_recognition_11.workflow import ThresholdConfiguration

from kolena.workflow import AxisConfig
from kolena.workflow import BarPlot
from kolena.workflow import Curve
from kolena.workflow import CurvePlot
from kolena.workflow import EvaluationResults
from kolena.workflow import Plot
from kolena.workflow import TestCases
from kolena.workflow.metrics import f1_score
from kolena.workflow.metrics import iou
from kolena.workflow.metrics import precision
from kolena.workflow.metrics import recall


def compute_per_sample(
    ground_truth: GroundTruth,
    inference: Inference,
    threshold: float,
    configuration: ThresholdConfiguration,
    test_sample: TestSample = None,
) -> TestSampleMetrics:
    # Stage 1: Detection
    iou_value = (
        iou(ground_truth.bbox, inference.bbox)
        if (ground_truth.bbox is not None and inference.bbox is not None)
        else 0.0
    )
    bbox_fte, tp, fp = False, False, False
    if iou_value != 0.0:
        tp = iou_value >= configuration.iou_threshold
        fp = iou_value < configuration.iou_threshold
    else:
        bbox_fte = True

    # Stage 2: Keypoints

    keypoint_fta = False
    mse, nmse = None, None
    Δ_nose, norm_Δ_nose = None, None
    Δ_left_eye, norm_Δ_left_eye = None, None
    Δ_right_eye, norm_Δ_right_eye = None, None
    Δ_left_mouth, norm_Δ_left_mouth = None, None
    Δ_right_mouth, norm_Δ_right_mouth = None, None
    if inference.keypoints is not None:
        normalization_factor = ground_truth.normalization_factor
        Δ_nose, norm_Δ_nose = compute_distances(
            ground_truth.keypoints.points[0],
            inference.keypoints.points[0],
            normalization_factor,
        )
        Δ_left_eye, norm_Δ_left_eye = compute_distances(
            ground_truth.keypoints.points[1],
            inference.keypoints.points[1],
            normalization_factor,
        )
        Δ_right_eye, norm_Δ_right_eye = compute_distances(
            ground_truth.keypoints.points[2],
            inference.keypoints.points[2],
            normalization_factor,
        )
        Δ_left_mouth, norm_Δ_left_mouth = compute_distances(
            ground_truth.keypoints.points[3],
            inference.keypoints.points[3],
            normalization_factor,
        )
        Δ_right_mouth, norm_Δ_right_mouth = compute_distances(
            ground_truth.keypoints.points[4],
            inference.keypoints.points[4],
            normalization_factor,
        )
        distances = np.array([Δ_left_eye, Δ_right_eye, Δ_nose, Δ_left_mouth, Δ_right_mouth])
        mse, nmse = calculate_mse_nmse(distances, normalization_factor)

    if not bbox_fte and (inference.keypoints is None or nmse >= configuration.nmse_threshold):
        keypoint_fta = True

    # Stage 3: Recognition
    pair_samples = list()
    count_fnm, count_fm, count_tm, count_tnm = 0, 0, 0, 0
    for i, (is_same, similarity) in enumerate(zip(ground_truth.matches, inference.similarities)):
        is_match, is_false_match, is_false_non_match, failure_to_enroll = False, False, False, False
        if similarity is None:
            if is_same:
                is_false_non_match = True
            failure_to_enroll = True
        elif similarity > threshold:  # match
            if is_same:
                is_match = True
            else:
                is_false_match = True
        else:  # no match
            if is_same:
                is_false_non_match = True

        pair_sample = PairSample(
            locator=test_sample.pairs[i].locator if test_sample else "",
            is_match=is_match,
            is_false_match=is_false_match,
            is_false_non_match=is_false_non_match,
            failure_to_enroll=failure_to_enroll,
            similarity=similarity,
        )
        pair_samples.append(pair_sample)
        count_fnm += is_false_non_match
        count_fm += is_false_match
        count_tm += is_match
        count_tnm += not is_false_non_match and not is_false_match and not is_match

    return TestSampleMetrics(
        pair_samples=pair_samples,
        count_FNM=count_fnm,
        count_FM=count_fm,
        count_TM=count_tm,
        count_TNM=count_tnm,
        similarity_threshold=threshold,
        bbox_IoU=iou_value if iou_value is not None else 0.0,
        bbox_TP=[inference.bbox] if tp and inference.bbox is not None else [],
        bbox_FP=[inference.bbox] if fp and inference.bbox is not None else [],
        bbox_FN=[inference.bbox] if (not tp and not fp) and inference.bbox is not None else [],
        bbox_has_TP=tp,
        bbox_has_FP=fp,
        bbox_has_FN=(not tp and not fp),
        bbox_failure_to_enroll=bbox_fte,
        keypoint_MSE=mse,
        keypoint_NMSE=nmse,
        keypoint_Δ_nose=Δ_nose,
        keypoint_Δ_left_eye=Δ_left_eye,
        keypoint_Δ_right_eye=Δ_right_eye,
        keypoint_Δ_left_mouth=Δ_left_mouth,
        keypoint_Δ_right_mouth=Δ_right_mouth,
        keypoint_norm_Δ_nose=norm_Δ_nose,
        keypoint_norm_Δ_left_eye=norm_Δ_left_eye,
        keypoint_norm_Δ_right_eye=norm_Δ_right_eye,
        keypoint_norm_Δ_left_mouth=norm_Δ_left_mouth,
        keypoint_norm_Δ_right_mouth=norm_Δ_right_mouth,
        keypoint_failure_to_align=keypoint_fta,
    )


def compute_per_bbox_case(
    ground_truths: List[GroundTruth],
    metrics: List[TestSampleMetrics],
) -> PerBBoxMetrics:
    tp = np.sum([tsm.bbox_has_TP for tsm in metrics])
    fp = np.sum([tsm.bbox_has_FP for tsm in metrics])
    fn = np.sum([tsm.bbox_has_FN for tsm in metrics])

    return PerBBoxMetrics(
        Label="Bounding Box Detection",
        Total=np.sum([gt.bbox is not None for gt in ground_truths]),
        FTE=np.sum([tsm.bbox_failure_to_enroll for tsm in metrics]),
        AvgIoU=np.mean([tsm.bbox_IoU for tsm in metrics]),
        Precision=precision(tp, fp),
        Recall=recall(tp, fn),
        F1=f1_score(tp, fp, fn),
        TP=tp,
        FP=fp,
        FN=fn,
    )


def compute_per_keypoint_case(ground_truths: List[GroundTruth], metrics: List[TestSampleMetrics]) -> PerKeypointMetrics:
    def process_metric(metric):
        return metric if metric is not None else 0.0

    return PerKeypointMetrics(
        Label="Keypoints Detection",
        Total=np.sum([gt.keypoints is not None for gt in ground_truths]),
        FTA=np.sum([tsm.keypoint_failure_to_align for tsm in metrics]),
        MSE=np.nanmean([process_metric(tsm.keypoint_MSE) for tsm in metrics]),
        NMSE=np.nanmean([process_metric(tsm.keypoint_NMSE) for tsm in metrics]),
        AvgΔNose=np.nanmean([process_metric(tsm.keypoint_Δ_nose) for tsm in metrics]),
        AvgΔLeftEye=np.nanmean([process_metric(tsm.keypoint_Δ_left_eye) for tsm in metrics]),
        AvgΔRightEye=np.nanmean([process_metric(tsm.keypoint_Δ_right_eye) for tsm in metrics]),
        AvgΔLeftMouth=np.nanmean([process_metric(tsm.keypoint_Δ_left_mouth) for tsm in metrics]),
        AvgΔRightMouth=np.nanmean([process_metric(tsm.keypoint_Δ_right_mouth) for tsm in metrics]),
        AvgNormΔNose=np.nanmean([process_metric(tsm.keypoint_norm_Δ_nose) for tsm in metrics]),
        AvgNormΔLeftEye=np.nanmean([process_metric(tsm.keypoint_norm_Δ_left_eye) for tsm in metrics]),
        AvgNormΔRightEye=np.nanmean([process_metric(tsm.keypoint_norm_Δ_right_eye) for tsm in metrics]),
        AvgNormΔLeftMouth=np.nanmean([process_metric(tsm.keypoint_norm_Δ_left_mouth) for tsm in metrics]),
        AvgNormΔRightMouth=np.nanmean([process_metric(tsm.keypoint_norm_Δ_right_mouth) for tsm in metrics]),
    )


def compute_test_case_metrics(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    metrics: List[TestSampleMetrics],
    baseline_fnmr: float,
) -> TestCaseMetrics:
    n_genuine_pairs, n_imposter_pairs, n_fm, n_fnm, n_pair_failures, n_fte = compute_pair_metrics(
        test_samples, ground_truths, metrics
    )

    bbox_metrics = compute_per_bbox_case(ground_truths, metrics)
    keypoint_metrics = compute_per_keypoint_case(ground_truths, metrics)

    return TestCaseMetrics(
        TotalPairs=n_genuine_pairs + n_imposter_pairs,
        nGenuinePairs=n_genuine_pairs,
        nImposterPairs=n_imposter_pairs,
        FM=n_fm,
        FMR=n_fm / n_imposter_pairs,
        FNM=n_fnm,
        FNMR=n_fnm / n_genuine_pairs,
        ΔFNMR=(n_fnm / n_genuine_pairs) - baseline_fnmr,
        FTE=n_fte,
        FTER=n_fte / len(test_samples),
        PairFailures=n_pair_failures,
        PairFailureRate=n_pair_failures / (n_genuine_pairs + n_imposter_pairs),
        PerBBoxMetrics=[bbox_metrics],
        PerKeypointMetrics=[keypoint_metrics],
    )


def compute_test_case_plots(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    metrics: List[TestSampleMetrics],
    configuration: ThresholdConfiguration,
) -> Optional[List[Plot]]:
    plots = []

    ### Plots for BBox ###
    tp = np.sum([tsm.bbox_has_TP for tsm in metrics])
    fp = np.sum([tsm.bbox_has_FP for tsm in metrics])
    fn = np.sum([tsm.bbox_has_FN for tsm in metrics])

    plots.append(
        BarPlot(
            title="Bounding Box Detection: Detection Histogram",
            x_label="Detection Outcomes",
            y_label="Count",
            labels=["TP", "FP", "FN"],
            values=[tp, fp, fn],
        ),
    )
    plots.append(create_iou_histogram(metrics))

    ### Plots for Keypoints ###
    def compute_failure_rate(nmse_threshold: float, nmses: List[Optional[float]]) -> float:
        total = sum(1 for nm in nmses if nm is not None and nm > nmse_threshold)
        return total / len(nmses) if len(nmses) > 0 else 0

    nmses = [mts.keypoint_NMSE for mts in metrics]
    x = np.linspace(0, 0.5, 251).tolist()
    y = [compute_failure_rate(x_value, nmses) for x_value in x]
    plots.append(
        CurvePlot(
            title="Keypoint Detection: Alignment Failure Rate vs. NMSE Threshold",
            x_label="NMSE Threshold",
            y_label="Alignment Failure Rate",
            curves=[Curve(label="NMSE", x=x, y=y)],
        ),
    )

    ### Plots for Face Recognition ###
    FMR_lower = -4
    FMR_upper = -1
    baseline_fmr_x = list(np.logspace(FMR_lower, FMR_upper, 50))

    thresholds = [compute_threshold(test_samples, ground_truths, inferences, fmr) for fmr in baseline_fmr_x]

    fnmr_y = list()
    fmr_y = list()

    for threshold in thresholds:
        tsm_for_one_threshold = [
            compute_per_sample(gt, inf, threshold, configuration) for gt, inf in zip(ground_truths, inferences)
        ]

        n_genuine, n_imposter, n_fm, n_fnm, _, _ = compute_pair_metrics(
            test_samples, ground_truths, tsm_for_one_threshold
        )
        fnmr_y.append(n_fnm / n_genuine)
        fmr_y.append(n_fm / n_imposter)

    plots.append(
        CurvePlot(
            title="Recognition: Test Case FNMR vs. Baseline FMR",
            x_label="Baseline False Match Rate",
            y_label="Test Case False Non-Match Rate (%)",
            x_config=AxisConfig(type="log"),
            curves=[Curve(x=baseline_fmr_x, y=fnmr_y, extra=dict(Threshold=thresholds))],
        ),
    )

    plots.append(
        CurvePlot(
            title="Recognition: Test Case FMR vs. Baseline FMR",
            x_label="Baseline False Match Rate",
            y_label="Test Case False Match Rate",
            x_config=AxisConfig(type="log"),
            y_config=AxisConfig(type="log"),
            curves=[Curve(x=baseline_fmr_x, y=fmr_y, extra=dict(Threshold=thresholds))],
        ),
    )

    plots.append(create_similarity_histogram(test_samples, ground_truths, inferences))

    return plots


def compute_test_suite_metrics(
    metrics: List[TestSampleMetrics],
    fm: int,
    fnm: int,
    fnmr: float,
    threshold: float,
) -> TestSuiteMetrics:
    return TestSuiteMetrics(
        Threshold=threshold,
        FM=fm,
        FNM=fnm,
        FNMR=fnmr,
        TotalFTE=np.sum([tsm.bbox_failure_to_enroll or tsm.keypoint_failure_to_align for tsm in metrics]),
        TotalBBoxFTE=np.sum([tsm.bbox_failure_to_enroll for tsm in metrics]),
        TotalKeypointFTE=np.sum([tsm.keypoint_failure_to_align for tsm in metrics]),
    )


def evaluate_face_recognition_11(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
    configuration: ThresholdConfiguration,
) -> EvaluationResults:
    threshold = compute_threshold(test_samples, ground_truths, inferences, configuration.false_match_rate)

    # compute per-sample metrics for each test sample
    test_sample_metrics = [
        compute_per_sample(gt, inf, threshold, configuration, test_samples[i])
        for i, (gt, inf) in enumerate(zip(ground_truths, inferences))
    ]

    # compute aggregate metrics across all test cases using `test_cases.iter(...)`
    all_test_case_metrics: List[Tuple[TestCase, TestCaseMetrics]] = []
    all_test_case_plots: List[Tuple[TestCase, List[Plot]]] = []
    n_genuine, _, n_fm, n_fnm, _, _ = compute_pair_metrics(test_samples, ground_truths, test_sample_metrics)
    fnmr = n_fnm / n_genuine
    for test_case, ts_subset, gt, inf, tsm in test_cases.iter(
        test_samples,
        ground_truths,
        inferences,
        test_sample_metrics,
    ):
        all_test_case_metrics.append((test_case, compute_test_case_metrics(ts_subset, gt, tsm, fnmr)))
        all_test_case_plots.append(
            (test_case, compute_test_case_plots(ts_subset, gt, inf, tsm, configuration)),
        )

    test_suite_metrics = compute_test_suite_metrics(test_sample_metrics, n_fm, n_fnm, fnmr, threshold)

    return EvaluationResults(
        metrics_test_sample=list(zip(test_samples, test_sample_metrics)),
        metrics_test_case=all_test_case_metrics,
        plots_test_case=all_test_case_plots,
        metrics_test_suite=test_suite_metrics,
    )
