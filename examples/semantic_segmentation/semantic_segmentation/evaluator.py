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
from semantic_segmentation.utils import compute_score_distribution_plot
from semantic_segmentation.utils import download_mask
from semantic_segmentation.utils import upload_image
from semantic_segmentation.workflow import GroundTruth
from semantic_segmentation.workflow import Inference
from semantic_segmentation.workflow import TestCase
from semantic_segmentation.workflow import TestCaseMetric
from semantic_segmentation.workflow import TestSample
from semantic_segmentation.workflow import TestSampleMetric

from kolena.workflow import EvaluationResults
from kolena.workflow import Plot
from kolena.workflow import TestCases
from kolena.workflow.annotation import SegmentationMask


ResultMasks = Tuple[SegmentationMask, SegmentationMask, SegmentationMask]

BUCKET = "kolena-public-datasets"
DATASET = "coco-stuff-10k"


def compute_test_sample_metrics(
    gt: GroundTruth,
    inf: Inference,
    ts: TestSample,
):
    gt_mask = download_mask(gt.mask.locator)
    inf_mask = download_mask(inf.mask.locator)
    tp, fp, fn = _load_sample_result_masks(test_sample=ts, gt_mask=gt_mask, inf_mask=inf_mask)

    count_tps = 0
    count_fps = 0
    count_fns = 0
    rows, cols = gt_mask.shape
    inf_val = 1
    for x in range(0, rows):
        for y in range(0, cols):
            if gt_mask[x, y] == 1 and inf_mask[x, y] == inf_val:
                count_tps += 1

            if gt_mask[x, y] != 1 and inf_mask[x, y] == inf_val:
                count_fps += 1

            if gt_mask[x, y] == 1 and inf_mask[x, y] != inf_val:
                count_fns += 1

    precision = count_tps / (count_tps + count_fps) if (count_tps + count_fps) > 0 else 0
    recall = count_tps / (count_tps + count_fns) if (count_tps + count_fns) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

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
    precision = count_tps / (count_tps + count_fps) if (count_tps + count_fps) > 0 else 0
    recall = count_tps / (count_tps + count_fns) if (count_tps + count_fns) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return TestCaseMetric(
        Precision=precision,
        Recall=recall,
        F1=f1,
    )


def compute_test_case_plots(
    test_samples: List[TestSample],
    inferences: List[Inference],
    metrics: List[TestSampleMetric],
) -> Optional[List[Plot]]:
    plots: List[Plot] = [
        compute_score_distribution_plot("F1", metrics, (0, 1, 101)),
    ]

    return [plot for plot in plots if plot is not None]


def _load_sample_result_masks(
    test_sample: TestSample,
    gt_mask: np.ndarray,
    inf_mask: np.ndarray,
) -> ResultMasks:
    def upload_result_mask(category: str, mask: np.ndarray) -> SegmentationMask:
        model_name = "pspnet_r101-d8_4xb4-40k_coco-stuff10k-512x512"
        locator = f"s3://{BUCKET}/{DATASET}/results/{model_name}/{category}/{test_sample.metadata['basename']}.png"
        upload_image(locator, mask)
        return SegmentationMask(locator=locator, labels={1: "person"})

    tp = upload_result_mask("TP", np.where(gt_mask != inf_mask, 0, inf_mask))
    fp = upload_result_mask("FP", np.where(gt_mask == inf_mask, 0, inf_mask))
    fn = upload_result_mask("FN", np.where(gt_mask == inf_mask, 0, gt_mask))
    return tp, fp, fn


def evaluate_semantic_segmentation(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
) -> EvaluationResults:
    test_sample_metrics = [
        compute_test_sample_metrics(gt, inf, ts) for gt, inf, ts in zip(ground_truths, inferences, test_samples)
    ]

    # compute aggregate metrics across all test cases using `test_cases.iter(...)`
    all_test_case_metrics: List[Tuple[TestCase, TestCaseMetric]] = []
    all_test_case_plots: List[Tuple[TestCase, List[Plot]]] = []
    for test_case, ts, gt, inf, tsm in test_cases.iter(test_samples, ground_truths, inferences, test_sample_metrics):
        all_test_case_metrics.append((test_case, compute_test_case_metrics(tsm)))
        all_test_case_plots.append((test_case, compute_test_case_plots(ts, inf, tsm)))

    # if desired, compute and add `plots_test_case` and `metrics_test_suite` to this `EvaluationResults`
    return EvaluationResults(
        metrics_test_sample=list(zip(test_samples, test_sample_metrics)),
        metrics_test_case=all_test_case_metrics,
        plots_test_case=all_test_case_plots,
    )