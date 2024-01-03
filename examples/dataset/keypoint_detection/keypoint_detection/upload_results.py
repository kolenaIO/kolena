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

import pandas as pd
from keypoint_detection.metrics import compute_metrics
from keypoint_detection.model import infer_random
from keypoint_detection.model import infer_retinaface
from tqdm import tqdm

import kolena
from kolena._experimental.dataset import fetch_dataset
from kolena._experimental.dataset import test


def run(args: Namespace) -> int:
    kolena.initialize(verbose=True)
    infer = infer_retinaface if args.model == "RetinaFace" else infer_random
    df = fetch_dataset(args.dataset)

    results = []
    for record in tqdm(df.itertuples(), total=len(df)):
        bboxes, faces = infer(record)
        metrics = compute_metrics(record.face, faces, record.normalization_factor)
        results.append(dict(locator=record.locator, raw_bboxes=bboxes, raw_faces=faces, **metrics))

    df_results = pd.DataFrame.from_records(results)
    test(args.dataset, args.model, df_results)
    return 0


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("model", type=str, choices=["RetinaFace", "random"], help="Name of model to test.")
    ap.add_argument("dataset", nargs="?", default="300-W", help="Name of dataset to use for testing.")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
