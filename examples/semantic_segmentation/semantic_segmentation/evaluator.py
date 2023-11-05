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
import os
from typing import Iterable
from typing import List
from typing import Tuple

import numpy as np
from semantic_segmentation.constants import DATASET
from semantic_segmentation.data_loader import DataLoader
from semantic_segmentation.data_loader import ResultMasks
from semantic_segmentation.utils import compute_precision_recall_f1
from semantic_segmentation.utils import compute_sklearn_arrays
from semantic_segmentation.workflow import GroundTruth
from semantic_segmentation.workflow import Inference
from semantic_segmentation.workflow import SegmentationConfiguration
from semantic_segmentation.workflow import TestCase
from semantic_segmentation.workflow import TestCaseMetric
from semantic_segmentation.workflow import TestSample
from semantic_segmentation.workflow import TestSampleMetric

from kolena._experimental.object_detection.utils import compute_average_precision
from kolena._utils.log import info
from kolena.workflow import EvaluationResults
from kolena.workflow import TestCases
from kolena.workflow.metrics import f1_score as compute_f1_score
from kolena.workflow.metrics import precision as compute_precision
from kolena.workflow.metrics import recall as compute_recall
from kolena.workflow.plot import Curve
from kolena.workflow.plot import CurvePlot


def load_data(
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
) -> Tuple[Iterable[np.ndarray], Iterable[np.ndarray]]:
    info("downloading bit masks...")
    data_loader = DataLoader()
    return zip(*data_loader.download_masks(ground_truths, inferences))


def apply_threshold(
    inf_probs: List[np.ndarray],
    threshold: float,
) -> List[np.ndarray]:
    inf_masks = []
    for inf_prob in inf_probs:
        inf_mask = np.zeros_like(inf_prob)
        inf_mask[inf_prob >= threshold] = 1
        inf_masks.append(inf_mask)

    return inf_masks


def compute_image_metrics(result_masks: ResultMasks) -> TestSampleMetric:
    tp_result_mask, fp_result_mask, fn_result_mask = result_masks

    count_tps = tp_result_mask.count
    count_fps = fp_result_mask.count
    count_fns = fn_result_mask.count

    precision = compute_precision(count_tps, count_fps)
    recall = compute_recall(count_tps, count_fns)
    f1 = compute_f1_score(count_tps, count_fps, count_fns)

    return TestSampleMetric(
        TP=tp_result_mask.mask,
        FP=fp_result_mask.mask,
        FN=fn_result_mask.mask,
        Precision=precision,
        Recall=recall,
        F1=f1,
        CountTP=count_tps,
        CountFP=count_fps,
        CountFN=count_fns,
    )


def compute_test_sample_metrics(
    test_samples: List[TestSample],
    gt_masks: List[np.ndarray],
    inf_masks: List[np.ndarray],
    threshold: float,
) -> List[TestSampleMetric]:
    out_bucket = os.environ["KOLENA_OUT_BUCKET"]
    model_name = os.environ["KOLENA_MODEL_NAME"]
    locator_prefix = f"s3://{out_bucket}/{DATASET}/results/{model_name}/{threshold:.2f}"
    data_loader = DataLoader()

    info("uploading result masks...")
    result_masks = data_loader.upload_masks(locator_prefix, test_samples, gt_masks, inf_masks)
    return [compute_image_metrics(result_mask) for result_mask in result_masks]


def evaluate_test_case(
    gt: List[np.ndarray],
    inf: List[np.ndarray],
    tsm: List[TestSampleMetric],
    eval_level: int,
) -> Tuple[TestCaseMetric, List[CurvePlot]]:
    y_true, y_pred = compute_sklearn_arrays(gt, inf)
    thresholds = list(np.linspace(0.0, 1.0, eval_level))
    precisions = []
    recalls = []
    f1s = []
    for t in thresholds:
        precision, recall, f1 = compute_precision_recall_f1(y_true, y_pred, t)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    f1_curve = Curve(x=thresholds, y=f1s, extra=dict(Precision=precisions, Recall=recalls))
    pr_curve = Curve(
        x=recalls[1:-1],
        y=precisions[1:-1],
        extra=dict(F1=f1s[1:-1], Threshold=thresholds[1:-1]),
    )
    test_case_metrics = compute_test_case_metrics(pr_curve, tsm)
    test_case_plots = [
        CurvePlot(
            title="F1 vs. Confidence Threshold",
            x_label="Confidence Threshold",
            y_label="F1",
            curves=[f1_curve],
        ),
        CurvePlot(
            title="Precision vs. Recall",
            x_label="Recall",
            y_label="Precision",
            curves=[pr_curve],
        ),
    ]
    return test_case_metrics, test_case_plots


def compute_test_case_metrics(
    pr_curve: Curve,
    metrics: List[TestSampleMetric],
) -> TestCaseMetric:
    count_tps = sum(metric.CountTP for metric in metrics)
    count_fps = sum(metric.CountFP for metric in metrics)
    count_fns = sum(metric.CountFN for metric in metrics)
    precision = compute_precision(count_tps, count_fps)
    recall = compute_recall(count_tps, count_fns)
    f1 = compute_f1_score(count_tps, count_fps, count_fns)

    return TestCaseMetric(
        Precision=precision,
        Recall=recall,
        F1=f1,
        AP=compute_average_precision(list(pr_curve.y), list(pr_curve.x)),
    )


def evaluate_semantic_segmentation(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
    configuration: SegmentationConfiguration,
) -> EvaluationResults:
    gt_masks, inf_probs = load_data(ground_truths, inferences)
    inf_masks = apply_threshold(inf_probs, configuration.threshold)

    test_sample_metrics = compute_test_sample_metrics(test_samples, gt_masks, inf_masks, configuration.threshold)

    # compute aggregate metrics across all test cases using `test_cases.iter(...)`
    all_test_case_metrics: List[Tuple[TestCase, TestCaseMetric]] = []
    all_test_case_plots: List[Tuple[TestCase, List[CurvePlot]]] = []
    for test_case, _, gt, inf, tsm in test_cases.iter(test_samples, gt_masks, inf_probs, test_sample_metrics):
        info(f"computing {test_case.name} test case metrics")
        test_case_metrics, test_case_plots = evaluate_test_case(gt, inf, tsm, configuration.eval_level)
        all_test_case_metrics.append((test_case, test_case_metrics))
        all_test_case_plots.append((test_case, test_case_plots))

    # if desired, compute and add `plots_test_case` and `metrics_test_suite` to this `EvaluationResults`
    return EvaluationResults(
        metrics_test_sample=list(zip(test_samples, test_sample_metrics)),
        metrics_test_case=all_test_case_metrics,
        plots_test_case=all_test_case_plots,
    )
