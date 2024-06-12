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
import json
from argparse import ArgumentParser
from argparse import Namespace

import pandas as pd
import s3fs
from face_recognition_11.constants import BUCKET
from face_recognition_11.constants import DATASET
from face_recognition_11.constants import TASK
from face_recognition_11.metrics import compute_pairwise_recognition_metrics
from face_recognition_11.metrics import compute_recognition_threshold

from kolena.dataset import upload_results


def run(args: Namespace) -> None:
    df_results = pd.read_csv(f"s3://{BUCKET}/{DATASET}/{TASK}/results/raw/{args.model}_{args.detector}.csv")
    df_results = df_results.drop_duplicates(["locator_1", "locator_2"])
    fs = s3fs.S3FileSystem()
    with fs.open(f"{BUCKET}/{DATASET}/{TASK}/results/raw/{args.model}_{args.detector}.config.json", "rb") as f:
        eval_config = json.load(f)

    similarity_threshold = compute_recognition_threshold(df_results, eval_config["false_match_rate"])
    df_results["metrics"] = df_results.apply(
        lambda record: compute_pairwise_recognition_metrics(record.is_match, record.similarity, similarity_threshold),
        axis=1,
    )
    df_results = df_results[["locator_1", "locator_2", "similarity", "metrics"]]

    upload_results(args.dataset, f"{args.model}_{args.detector}", [(eval_config, df_results)])


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--model", type=str, choices=["vgg-face", "facenet512"], help="Name of FR model to test.")
    ap.add_argument("--detector", type=str, choices=["mtcnn", "dlib"], help="Name of detector backend to test.")
    ap.add_argument(
        "--dataset",
        type=str,
        default=f"{DATASET} [{TASK}]",
        help="Optionally specify a custom dataset name to test.",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
