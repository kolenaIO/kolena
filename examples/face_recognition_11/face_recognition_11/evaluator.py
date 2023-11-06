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
from face_recognition_11.utils import create_similarity_histogram
from face_recognition_11.workflow import ThresholdConfiguration
from face_recognition_11.workflow import GroundTruth
from face_recognition_11.workflow import Inference
from face_recognition_11.workflow import TestCase
from face_recognition_11.workflow import TestCaseMetrics
from face_recognition_11.workflow import TestSample
from face_recognition_11.workflow import TestSampleMetrics
from face_recognition_11.workflow import TestSuiteMetrics
from face_recognition_11.workflow import PerBBoxMetrics
from face_recognition_11.workflow import PerKeypointMetrics
from face_recognition_11.workflow import KeypointSample
from face_recognition_11.utils import compute_distances, calculate_mse_nmse

from kolena.workflow import AxisConfig
from kolena.workflow import Curve
from kolena.workflow import CurvePlot
from kolena.workflow import EvaluationResults
from kolena.workflow import Plot
from kolena.workflow import TestCases
from kolena.workflow.metrics import precision, recall, f1_score, iou


def compute_threshold(ground_truths, inferences, fmr: float, eps: float = 1e-9) -> float:
    total_matches = []
    total_similarities = []
    for sublist in ground_truths:
        for item in sublist.matches:
            total_matches.append(item)

    for sublist in inferences:
        for item in sublist.similarities:
            total_similarities.append(item)

    imposter_scores = sorted(
        [
            similarity if similarity else 0.0
            for match, similarity in zip(total_matches, total_similarities)
            if not match
        ],
        reverse=True,
    )
    threshold_idx = int(round(fmr * len(imposter_scores)) - 1)
    threshold = imposter_scores[threshold_idx] - eps
    return threshold


def compute_per_sample(
    ground_truth: GroundTruth,
    inference: Inference,
    threshold: float,
) -> TestSampleMetrics:
    m, fm, fnm, fte = [], [], [], []
    mse, Δ_nose, Δ_left_eye, Δ_right_eye, Δ_left_mouth, Δ_right_mouth = None, None, None, None, None, None

    if inference.keypoints:
        Δ_nose = compute_distances(ground_truth.keypoints.points[0], inference.keypoints.points[0])
        Δ_left_eye = compute_distances(ground_truth.keypoints.points[1], inference.keypoints.points[1])
        Δ_right_eye = compute_distances(ground_truth.keypoints.points[2], inference.keypoints.points[2])
        Δ_left_mouth = compute_distances(ground_truth.keypoints.points[3], inference.keypoints.points[3])
        Δ_right_mouth = compute_distances(ground_truth.keypoints.points[4], inference.keypoints.points[4])
        distances = np.array([Δ_left_eye, Δ_right_eye, Δ_nose, Δ_left_mouth, Δ_right_mouth])
        mse = calculate_mse_nmse(distances)

    for is_same, similarity in zip(ground_truth.matches, inference.similarities):
        if similarity is None:
            if is_same:
                fnm.append(True)
            fte.append(True)
            continue

        if similarity > threshold:  # match
            if is_same:
                m.append(True)
            else:
                fm.append(True)
        else:  # no match
            if is_same:
                fnm.append(True)

    return TestSampleMetrics(
        is_match=m,
        is_false_match=fm,
        is_false_non_match=fnm,
        failure_to_enroll=fte,
        mse=mse,
        Δ_nose=Δ_nose,
        Δ_left_eye=Δ_left_eye,
        Δ_right_eye=Δ_right_eye,
        Δ_left_mouth=Δ_left_mouth,
        Δ_right_mouth=Δ_right_mouth,
    )


def compute_per_bbox_case(
    ground_truths: List[GroundTruth], inferences: List[Inference], configuration: ThresholdConfiguration
) -> PerBBoxMetrics:
    n = np.sum([gt.bbox is not None for gt in ground_truths])
    fte = np.sum([inf.bbox is None for inf in inferences])

    ious = []
    for gt, inf in zip(ground_truths, inferences):
        if gt.bbox and inf.bbox:
            ious.append(iou(gt.bbox, inf.bbox))

    tp = np.sum([iou >= configuration.iou_threshold for iou in ious])
    fp = np.sum([iou < configuration.iou_threshold for iou in ious])
    fn = np.sum([iou == -1 for iou in ious])

    return PerBBoxMetrics(
        Label="Bounding Box Detection",
        Total=n,
        FTE=fte,
        AvgIoU=np.mean(ious),
        Precision=precision(tp, fp),
        Recall=recall(tp, fn),
        F1=f1_score(tp, fp, fn),
        TP=tp,
        FP=fp,
        FN=fn,
    )


def compute_per_keypoint_case(
    ground_truths: List[GroundTruth], inferences: List[Inference], metrics: List[TestSampleMetrics]
) -> PerKeypointMetrics:
    n = np.sum([gt.keypoints is not None for gt in ground_truths])
    fte = np.sum([inf.keypoints is None for inf in inferences])
    mse = np.mean([tsm.mse for tsm in metrics])
    avg_Δ_nose = np.mean([tsm.Δ_nose for tsm in metrics])
    avg_Δ_left_eye = np.mean([tsm.Δ_left_eye for tsm in metrics])
    avg_Δ_right_eye = np.mean([tsm.Δ_right_eye for tsm in metrics])
    avg_Δ_left_mouth = np.mean([tsm.Δ_left_mouth for tsm in metrics])
    avg_Δ_right_mouth = np.mean([tsm.Δ_right_mouth for tsm in metrics])

    return PerKeypointMetrics(
        Label="Keypoints Detection",
        Total=n,
        FTE=fte,
        MSE=mse,
        AvgΔNose=avg_Δ_nose,
        AvgΔLeftEye=avg_Δ_left_eye,
        AvgΔRightEye=avg_Δ_right_eye,
        AvgΔLeftMouth=avg_Δ_left_mouth,
        AvgΔRightMouth=avg_Δ_right_mouth,
    )


def compute_test_case_metrics(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    metrics: List[TestSampleMetrics],
    baseline_fnmr: float,
    configuration: ThresholdConfiguration,
) -> TestCaseMetrics:
    n_genuine_pairs = np.sum([np.sum(gt.matches) for gt in ground_truths])
    n_imposter_pairs = np.sum([np.sum(np.invert(gt.matches)) for gt in ground_truths])

    n_fm = np.sum([np.sum(metric.is_false_match) for metric in metrics])
    n_fnm = np.sum([np.sum(metric.is_false_non_match) for metric in metrics])
    n_pair_failures = np.sum([np.sum(metric.failure_to_enroll) for metric in metrics])
    n_fte = np.sum([inf.keypoints is None for inf in inferences])

    bbox_metrics = compute_per_bbox_case(ground_truths, inferences, configuration)
    keypoint_metrics = compute_per_keypoint_case(ground_truths, inferences, metrics)

    return TestCaseMetrics(
        nImages=len(test_samples),
        nGenuinePairs=n_genuine_pairs,
        nImposterPairs=n_imposter_pairs,
        FM=n_fm,
        FMR=(n_fm / n_imposter_pairs) * 100,
        FNM=n_fnm,
        FNMR=(n_fnm / n_genuine_pairs) * 100,
        ΔFNMR=(n_fnm / n_genuine_pairs) * 100 - baseline_fnmr,
        FTE=n_fte,
        FTER=(n_fte / len(test_samples)) * 100,
        PairFailures=n_pair_failures,
        PairFailureRate=(n_pair_failures / (n_genuine_pairs + n_imposter_pairs)) * 100,
        PerBBoxMetrics=[bbox_metrics],
        PerKeypointMetrics=[keypoint_metrics],
    )


def compute_test_case_plots(ground_truths: List[GroundTruth], inferences: List[Inference]) -> Optional[List[Plot]]:
    FMR_lower = -6
    FMR_upper = -1
    baseline_fmr_x = list(np.logspace(FMR_lower, FMR_upper, 50))

    thresholds = [compute_threshold(ground_truths, inferences, fmr) for fmr in baseline_fmr_x]

    n_genuine_pairs = np.sum([np.sum(gt.matches) for gt in ground_truths])
    n_imposter_pairs = np.sum([np.sum(np.invert(gt.matches)) for gt in ground_truths])

    fnmr_y = list()
    fmr_y = list()

    for threshold in thresholds:
        tsm_for_one_threshold = [compute_per_sample(gt, inf, threshold) for gt, inf in zip(ground_truths, inferences)]
        n_fm = np.sum(
            [np.sum(metric.is_false_match and not metric.failure_to_enroll) for metric in tsm_for_one_threshold]
        )
        n_fnm = np.sum(
            [np.sum(metric.is_false_non_match and not metric.failure_to_enroll) for metric in tsm_for_one_threshold],
        )
        fnmr_y.append((n_fnm / n_genuine_pairs) * 100)
        fmr_y.append(n_fm / n_imposter_pairs)

    curve_test_case_fnmr = CurvePlot(
        title="Test Case FNMR vs. Baseline FMR",
        x_label="Baseline False Match Rate",
        y_label="Test Case False Non-Match Rate (%)",
        x_config=AxisConfig(type="log"),
        curves=[Curve(x=baseline_fmr_x, y=fnmr_y, extra=dict(Threshold=thresholds))],
    )

    curve_test_case_fmr = CurvePlot(
        title="Test Case FMR vs. Baseline FMR",
        x_label="Baseline False Match Rate",
        y_label="Test Case False Match Rate",
        x_config=AxisConfig(type="log"),
        y_config=AxisConfig(type="log"),
        curves=[Curve(x=baseline_fmr_x, y=fmr_y, extra=dict(Threshold=thresholds))],
    )

    similarity_dist = create_similarity_histogram(ground_truths, inferences)

    return [similarity_dist, curve_test_case_fnmr, curve_test_case_fmr]


def compute_test_suite_metrics(baseline: TestCaseMetrics, threshold: float) -> TestSuiteMetrics:
    return TestSuiteMetrics(
        Threshold=threshold,
        FM=baseline.FM,
        FNM=baseline.FNM,
        FNMR=(baseline.FNM / baseline.nGenuinePairs) * 100,
    )


def evaluate_face_recognition_11(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
    configuration: ThresholdConfiguration,
) -> EvaluationResults:
    threshold = compute_threshold(ground_truths, inferences, configuration.false_match_rate)

    # compute per-sample metrics for each test sample
    test_sample_metrics = [compute_per_sample(gt, inf, threshold) for gt, inf in zip(ground_truths, inferences)]

    # compute aggregate metrics across all test cases using `test_cases.iter(...)`
    all_test_case_metrics: List[Tuple[TestCase, TestCaseMetrics]] = []
    all_test_case_plots: List[Tuple[TestCase, List[Plot]]] = []
    baseline = compute_test_case_metrics(test_samples, ground_truths, inferences, test_sample_metrics, 0, configuration)
    baseline_fnmr = baseline.FNMR
    for test_case, ts, gt, inf, tsm in test_cases.iter(test_samples, ground_truths, inferences, test_sample_metrics):
        all_test_case_metrics.append(
            (test_case, compute_test_case_metrics(ts, gt, inf, tsm, baseline_fnmr, configuration))
        )
        all_test_case_plots.append((test_case, compute_test_case_plots(gt, inf)))

    test_suite_metrics = compute_test_suite_metrics(baseline, threshold)

    return EvaluationResults(
        metrics_test_sample=list(zip(test_samples, test_sample_metrics)),
        metrics_test_case=all_test_case_metrics,
        plots_test_case=all_test_case_plots,
        metrics_test_suite=test_suite_metrics,
    )
