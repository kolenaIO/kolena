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
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from semantic_segmentation.constants import BUCKET
from semantic_segmentation.constants import DATASET
from semantic_segmentation.data_loader import DataLoader
from semantic_segmentation.utils import upload_image
from semantic_segmentation.workflow import GroundTruth
from semantic_segmentation.workflow import Inference
from semantic_segmentation.workflow import Label
from semantic_segmentation.workflow import SegmentationConfiguration
from semantic_segmentation.workflow import TestCase
from semantic_segmentation.workflow import TestCaseMetric
from semantic_segmentation.workflow import TestSample
from semantic_segmentation.workflow import TestSampleMetric

from kolena._utils.log import progress_bar
from kolena.workflow import EvaluationResults
from kolena.workflow import Plot
from kolena.workflow import TestCases
from kolena.workflow.annotation import SegmentationMask
from kolena.workflow.metrics import f1_score as compute_f1_score
from kolena.workflow.metrics import precision as compute_precision
from kolena.workflow.metrics import recall as compute_recall


ResultMasks = Tuple[SegmentationMask, SegmentationMask, SegmentationMask]


def _upload_sample_result_masks(
    test_sample: TestSample,
    gt_mask: np.ndarray,
    inf_mask: np.ndarray,
    model_name: str,
) -> ResultMasks:
    # TODO: change the directory to include threshold
    def upload_result_mask(category: str, mask: np.ndarray) -> SegmentationMask:
        model_name = "pspnet_r101-d8_4xb4-40k_coco-stuff10k-512x512"
        locator = f"s3://{BUCKET}/{DATASET}/results/{model_name}/{category}/{test_sample.metadata['basename']}.png"
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

    return gt_masks, inf_probs


def compute_test_sample_metrics(
    ts: TestSample,
    gt_mask: np.ndarray,
    inf_mask: np.ndarray,
    configuration: SegmentationConfiguration,
):
    tp, fp, fn = _upload_sample_result_masks(
        test_sample=ts,
        gt_mask=gt_mask,
        inf_mask=inf_mask,
        model_name=configuration.model_name,
    )

    count_tps = 0
    count_fps = 0
    count_fns = 0
    rows, cols = gt_mask.shape
    for x in range(0, rows):
        for y in range(0, cols):
            if gt_mask[x, y] == 1 and inf_mask[x, y] == 1:
                count_tps += 1

            if gt_mask[x, y] != 1 and inf_mask[x, y] == 1:
                count_fps += 1

            if gt_mask[x, y] == 1 and inf_mask[x, y] != 1:
                count_fns += 1

    precision = compute_precision(count_tps, count_fps)
    recall = compute_recall(count_tps, count_fns)
    f1 = compute_f1_score(count_tps, count_fps, count_fns)

    return TestSampleMetric(
        TP=tp,
        FP=fp,
        FN=fn,
        Precision=precision,
        Recall=recall,
        F1=f1,
        CountTP=count_tps,
        CountFP=count_fps,
        CountFN=count_fns,
    )


def compute_test_case_metrics(
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
        AP=0.0,  # TODO: compute AP
    )


def compute_test_case_plots(
    test_samples: List[TestSample],
    inferences: List[Inference],
    metrics: List[TestSampleMetric],
) -> Optional[List[Plot]]:
    # TODO: complete this function
    plots: List[Plot] = []

    return [plot for plot in plots if plot is not None]


def compute_f1_optimal_threshold(
    gt_masks: List[np.ndarray],
    inf_probs: List[np.ndarray],
) -> float:
    # TODO complete this function
    return 0.5


def compute_threshold(
    configuration: SegmentationConfiguration,
    gt_masks: List[np.ndarray],
    inf_probs: List[np.ndarray],
) -> float:
    if configuration.threshold == "F1-Optimal":
        return compute_f1_optimal_threshold(gt_masks, inf_probs)
    else:
        return configuration.threshold


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


def evaluate_semantic_segmentation(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
    configuration: SegmentationConfiguration,
) -> EvaluationResults:
    gt_masks, inf_probs = load_data(test_samples, ground_truths, inferences)
    threshold = compute_threshold(configuration, gt_masks, inf_probs)
    inf_masks = apply_threshold(inf_probs, threshold)

    test_sample_metrics = [
        compute_test_sample_metrics(ts, gt, inf, configuration)
        for ts, gt, inf in zip(test_samples, gt_masks, inf_masks)
    ]

    # compute aggregate metrics across all test cases using `test_cases.iter(...)`
    all_test_case_metrics: List[Tuple[TestCase, TestCaseMetric]] = []
    all_test_case_plots: List[Tuple[TestCase, List[Plot]]] = []
    for test_case, ts, gt, inf, tsm in test_cases.iter(test_samples, gt_masks, inf_masks, test_sample_metrics):
        # compute_sklearn_arrays(gt_masks, inf_probs)
        # TODO: figure out a way to extract test case level masks
        all_test_case_metrics.append((test_case, compute_test_case_metrics(tsm)))
        all_test_case_plots.append((test_case, compute_test_case_plots(ts, inf, tsm)))

    # if desired, compute and add `plots_test_case` and `metrics_test_suite` to this `EvaluationResults`
    return EvaluationResults(
        metrics_test_sample=list(zip(test_samples, test_sample_metrics)),
        metrics_test_case=all_test_case_metrics,
        plots_test_case=all_test_case_plots,
    )
