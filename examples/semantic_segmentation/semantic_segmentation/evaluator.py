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
import math
import os
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from semantic_segmentation.constants import DATASET
from semantic_segmentation.data_loader import DataLoader
from semantic_segmentation.data_loader import ResultMasks
from semantic_segmentation.utils import compute_precision_recall_f1
from semantic_segmentation.utils import compute_sklearn_arrays
from semantic_segmentation.utils import upload_image
from semantic_segmentation.workflow import GroundTruth
from semantic_segmentation.workflow import Inference
from semantic_segmentation.workflow import Label
from semantic_segmentation.workflow import SegmentationConfiguration
from semantic_segmentation.workflow import TestCase
from semantic_segmentation.workflow import TestCaseMetric
from semantic_segmentation.workflow import TestSample
from semantic_segmentation.workflow import TestSampleMetric
from sklearn.metrics import average_precision_score

from kolena._utils.log import progress_bar
from kolena.workflow import EvaluationResults
from kolena.workflow import Plot
from kolena.workflow import TestCases
from kolena.workflow.annotation import SegmentationMask
from kolena.workflow.metrics import f1_score as compute_f1_score
from kolena.workflow.metrics import precision as compute_precision
from kolena.workflow.metrics import recall as compute_recall
from kolena.workflow.plot import Curve
from kolena.workflow.plot import CurvePlot


def _upload_sample_result_masks(
    test_sample: TestSample,
    gt_mask: np.ndarray,
    inf_mask: np.ndarray,
    locator_prefix: str,
) -> ResultMasks:
    def upload_result_mask(category: str, mask: np.ndarray) -> SegmentationMask:
        locator = f"{locator_prefix}/{category}/{test_sample.metadata['basename']}.png"
        upload_image(locator, mask)
        return SegmentationMask(locator=locator, labels=Label.as_label_map())

    tp = upload_result_mask("TP", np.where(gt_mask != inf_mask, 0, inf_mask))
    fp = upload_result_mask("FP", np.where(gt_mask == inf_mask, 0, inf_mask))
    fn = upload_result_mask("FN", np.where(gt_mask == inf_mask, 0, gt_mask))
    return tp, fp, fn


def load_data(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    data_loader = DataLoader()

    batch_size = 32
    n_batches = math.ceil(len(inferences) / batch_size)
    print(f"processing {len(inferences)} images in {n_batches} batches...")

    gt_masks = []
    inf_probs = []
    zipped_data = list(zip(test_samples, ground_truths, inferences))
    for batch in progress_bar(np.array_split(zipped_data, n_batches)):
        gt_masks_batch, inf_probs_batch = data_loader.load_batch(batch)
        gt_masks.extend(gt_masks_batch)
        inf_probs.extend(inf_probs_batch)

    print(f"finished loading {len(inferences)} ground truth masks and inferences")
    return gt_masks, inf_probs


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


def compute_image_metrics(gt_mask: np.ndarray, inf_mask: np.ndarray, result_masks: ResultMasks) -> TestSampleMetric:
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

    batch_size = 32
    n_batches = math.ceil(len(test_samples) / batch_size)
    print(f"computing {len(test_samples)} sample metrics in {n_batches} batches...")

    result_masks = []
    zipped_data = list(zip(test_samples, gt_masks, inf_masks))

    for start_index in progress_bar(range(0, len(test_samples), batch_size)):
        batch = zipped_data[start_index : min(len(test_samples), start_index + batch_size)]
        result_masks_batch = data_loader.upload_batch(locator_prefix, batch)
        result_masks.extend(result_masks_batch)

    print(f"finished uploading result masks for {len(test_samples)} samples")
    return [compute_image_metrics(gt, inf, result) for gt, inf, result in zip(gt_masks, inf_masks, result_masks)]


def compute_test_case_metrics(
    gt_masks: List[np.ndarray],
    inf_probs: List[np.ndarray],
    metrics: List[TestSampleMetric],
) -> TestCaseMetric:
    count_tps = sum(metric.CountTP for metric in metrics)
    count_fps = sum(metric.CountFP for metric in metrics)
    count_fns = sum(metric.CountFN for metric in metrics)
    precision = compute_precision(count_tps, count_fps)
    recall = compute_recall(count_tps, count_fns)
    f1 = compute_f1_score(count_tps, count_fps, count_fns)
    y_true, y_pred = compute_sklearn_arrays(gt_masks, inf_probs)

    return TestCaseMetric(
        Precision=precision,
        Recall=recall,
        F1=f1,
        AP=average_precision_score(y_true, y_pred),
    )


def compute_test_case_plots(
    gt_masks: List[np.ndarray],
    inf_probs: List[np.ndarray],
) -> Optional[List[Plot]]:
    y_true, y_pred = compute_sklearn_arrays(gt_masks, inf_probs)
    thresholds = list(np.linspace(0.0, 1.0, 41))

    precisions = []
    recalls = []
    f1s = []
    for t in thresholds:
        precision, recall, f1 = compute_precision_recall_f1(y_true, y_pred, t)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    f1_curve = Curve(x=thresholds, y=f1s, extra=dict(Precision=precisions, Recall=recalls))
    pr_curve = Curve(x=recalls[1:-1], y=precisions[1:-1], extra=dict(F1=f1s[1:-1], Threshold=thresholds[1:-1]))
    return [
        CurvePlot(
            title="F1 vs. Confidence Threshold",
            x_label="Confidence Threshold",
            y_label="F1",
            curves=[f1_curve],
        ),
        CurvePlot(title="Precision vs. Recall", x_label="Recall", y_label="Precision", curves=[pr_curve]),
    ]


def evaluate_semantic_segmentation(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
    configuration: SegmentationConfiguration,
) -> EvaluationResults:
    gt_masks, inf_probs = load_data(test_samples, ground_truths, inferences)
    inf_masks = apply_threshold(inf_probs, configuration.threshold)

    test_sample_metrics = compute_test_sample_metrics(test_samples, gt_masks, inf_masks, configuration.threshold)

    # compute aggregate metrics across all test cases using `test_cases.iter(...)`
    all_test_case_metrics: List[Tuple[TestCase, TestCaseMetric]] = []
    all_test_case_plots: List[Tuple[TestCase, List[Plot]]] = []
    for test_case, ts, gt, inf, tsm in test_cases.iter(test_samples, gt_masks, inf_probs, test_sample_metrics):
        print(f"computing {test_case.name} test case metrics")
        all_test_case_metrics.append((test_case, compute_test_case_metrics(gt, inf, tsm)))
        all_test_case_plots.append((test_case, compute_test_case_plots(gt, inf)))

    # if desired, compute and add `plots_test_case` and `metrics_test_suite` to this `EvaluationResults`
    return EvaluationResults(
        metrics_test_sample=list(zip(test_samples, test_sample_metrics)),
        metrics_test_case=all_test_case_metrics,
        plots_test_case=all_test_case_plots,
    )
