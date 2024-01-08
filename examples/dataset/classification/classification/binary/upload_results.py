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
from typing import Any
from typing import Dict

import pandas as pd
from classification.binary.constants import BUCKET
from classification.binary.constants import DATASET
from classification.binary.constants import ID_FIELDS

import kolena
from kolena.annotation import ScoredClassificationLabel
from kolena.dataset import download_dataset
from kolena.dataset import upload_results


POSITIVE_LABEL = "dog"
NEGATIVE_LABEL = "not dog"
MODELS = ["resnet50v2", "inceptionv3"]
EVAL_CONFIG = {"threshold": 0.5}


def compute_metrics(score: float, ground_truth_label: str) -> Dict[str, Any]:
    threshold = EVAL_CONFIG["threshold"]
    is_positive_prediction = score >= threshold
    classification_label = POSITIVE_LABEL if is_positive_prediction else NEGATIVE_LABEL
    is_positive_sample = ground_truth_label == classification_label

    return {
        "threshold": threshold,
        "is_correct": is_positive_prediction == is_positive_sample,
        "is_TP": is_positive_sample and is_positive_prediction,
        "is_FP": not is_positive_sample and is_positive_prediction,
        "is_FN": is_positive_sample and not is_positive_prediction,
        "is_TN": not is_positive_sample and not is_positive_prediction,
    }


def create_classification(score: float) -> ScoredClassificationLabel:
    threshold = EVAL_CONFIG["threshold"]
    label = POSITIVE_LABEL if score >= threshold else NEGATIVE_LABEL
    score = score if score >= threshold else 1 - score
    return ScoredClassificationLabel(label=label, score=score)


def run(args: Namespace) -> None:
    kolena.initialize(verbose=True)
    dataset_df = download_dataset(args.dataset)
    df_results = pd.read_csv(
        f"s3://{BUCKET}/{DATASET}/results/raw/{args.model}.csv",
        storage_options={"anon": True},
    )

    df_results = df_results.merge(dataset_df, how="left", on=ID_FIELDS)
    df_results["inference"] = df_results["prediction"].apply(lambda score: create_classification(score))
    eval_result = df_results.apply(lambda row: pd.Series(compute_metrics(row.prediction, row.label)), axis=1)
    df_results = pd.concat([df_results[ID_FIELDS], df_results["inference"], eval_result], axis=1)

    upload_results(args.dataset, args.model, [(EVAL_CONFIG, df_results)])


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument(
        "model",
        type=str,
        choices=MODELS,
        help="Name of the model to test.",
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
