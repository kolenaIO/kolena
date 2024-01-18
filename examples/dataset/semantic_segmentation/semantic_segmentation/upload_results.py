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
from argparse import ArgumentParser
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from typing import Dict
from typing import NamedTuple

import numpy as np
import pandas as pd
from semantic_segmentation.constants import BUCKET
from semantic_segmentation.constants import DATASET
from semantic_segmentation.constants import EVAL_CONFIG
from semantic_segmentation.constants import MODEL_NAME
from semantic_segmentation.utils import activation_map_locator_from_basename
from semantic_segmentation.utils import apply_threshold
from semantic_segmentation.utils import compute_result_masks
from semantic_segmentation.utils import inference_locator_from_basename
from semantic_segmentation.utils import load_data
from semantic_segmentation.utils import result_masks_locator_path
from semantic_segmentation.utils import upload_result_masks
from tqdm import tqdm

import kolena
from kolena.annotation import BitmapMask
from kolena.annotation import SegmentationMask
from kolena.asset import BinaryAsset
from kolena.dataset import download_dataset
from kolena.dataset import upload_results
from kolena.workflow.metrics import f1_score as compute_f1_score
from kolena.workflow.metrics import precision as compute_precision
from kolena.workflow.metrics import recall as compute_recall


class RecordType(NamedTuple):
    mask: SegmentationMask
    threshold: float
    basename: str
    locator: str
    inference_locator: str
    activation_map_locator: str
    result_masks_locator_prefix: str


def compute_metrics(record: RecordType) -> Dict[str, Any]:
    gt_mask, inf_proba = load_data(record.mask.locator, record.inference_locator)
    inf_mask = apply_threshold(inf_proba, record.threshold)
    tp_mask, fp_mask, fn_mask = compute_result_masks(gt_mask, inf_mask)
    tp_locator, fp_locator, fn_locator = upload_result_masks(
        tp_mask,
        fp_mask,
        fn_mask,
        record.result_masks_locator_prefix,
        record.basename,
    )

    count_tp = np.sum(tp_mask)
    count_fp = np.sum(fp_mask)
    count_fn = np.sum(fn_mask)

    return dict(
        locator=record.locator,
        threshold=record.threshold,
        inference=BinaryAsset(record.inference_locator),
        activation_map=BitmapMask(locator=record.activation_map_locator),
        TP=SegmentationMask(locator=tp_locator, labels={1: "PERSON"}),
        FP=SegmentationMask(locator=fp_locator, labels={1: "PERSON"}),
        FN=SegmentationMask(locator=fn_locator, labels={1: "PERSON"}),
        precision=compute_precision(count_tp, count_fp),
        recall=compute_recall(count_tp, count_fn),
        f1=compute_f1_score(count_tp, count_fp, count_fn),
        count_TP=count_tp,
        count_FP=count_fp,
        count_FN=count_fn,
    )


def run(args: Namespace) -> None:
    kolena.initialize(verbose=True)

    threshold = EVAL_CONFIG["threshold"]

    df_dataset = download_dataset(args.dataset)
    df_dataset["threshold"] = threshold
    df_dataset["result_masks_locator_prefix"] = result_masks_locator_path(
        args.write_bucket,
        DATASET,
        args.model,
        threshold,
    )
    df_dataset["inference_locator"] = df_dataset["basename"].apply(
        lambda x: inference_locator_from_basename(BUCKET, DATASET, args.model, x),
    )
    df_dataset["activation_map_locator"] = df_dataset["basename"].apply(
        lambda x: activation_map_locator_from_basename(BUCKET, DATASET, args.model, x),
    )

    pool = ThreadPoolExecutor(max_workers=32)
    results = list(tqdm(pool.map(compute_metrics, list(df_dataset.itertuples(index=False))), total=len(df_dataset)))

    df_results = pd.DataFrame.from_records(results)
    upload_results(args.dataset, MODEL_NAME[args.model], [(EVAL_CONFIG, df_results)])


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument(
        "model",
        type=str,
        default="pspnet_r101",
        nargs="?",
        choices=list(MODEL_NAME.keys()),
        help="Name of the model to test.",
    )
    ap.add_argument(
        "--write-bucket",
        type=str,
        required=True,
        help="Name of AWS S3 bucket with write access to upload result masks to.",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default=DATASET,
        help="Optionally specify a custom dataset name to test.",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
