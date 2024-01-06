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
from classification.binary.constants import NEGATIVE_LABEL
from classification.binary.constants import POSITIVE_LABEL

import kolena
from kolena.dataset import fetch_dataset
from kolena.dataset import upload_results
from kolena.workflow.annotation import ScoredClassificationLabel

MODELS = ["resnet50v2", "inceptionv3"]
EVAL_CONFIG = {"threshold": 0.5}
id_fields = ["locator"]


def metrics(score, ground_truth_label) -> Dict[str, Any]:
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


def to_kolena_inference(score) -> ScoredClassificationLabel:
    threshold = EVAL_CONFIG["threshold"]
    label = POSITIVE_LABEL if score >= threshold else NEGATIVE_LABEL
    score = score if score >= threshold else 1 - score
    return ScoredClassificationLabel(label=label, score=score)


def _upload_results(model_name: str, dataset: str) -> None:
    df_results = pd.read_csv(f"s3://{BUCKET}/{DATASET}/results/raw/{model_name}.csv")
    dataset_df = fetch_dataset(dataset)
    df_results = df_results.merge(dataset_df, how="left", on=id_fields)

    df_results["inference"] = df_results["prediction"].apply(lambda score: to_kolena_inference(score))

    eval_result = df_results.apply(lambda row: pd.Series(metrics(row.prediction, row.label)), axis=1)
    eval_result.columns = [
        "threshold",
        "is_correct",
        "is_TP",
        "is_FP",
        "is_FN",
        "is_TN",
    ]

    df_results = pd.concat([df_results[id_fields], df_results["inference"], eval_result], axis=1)
    upload_results(dataset, model_name, [(EVAL_CONFIG, df_results)])


def run(args: Namespace) -> None:
    kolena.initialize(verbose=True)
    for model_name in args.models:
        _upload_results(model_name, args.dataset)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        choices=MODELS,
        help="Name(s) of the models(s) to register.",
    )
    ap.add_argument(
        "--dataset",
        default=DATASET,
        help=f"Custom name for the {DATASET} dataset to test.",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
