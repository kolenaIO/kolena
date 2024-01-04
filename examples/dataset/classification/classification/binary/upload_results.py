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
from typing import List

import pandas as pd
from classification.binary.constants import BUCKET
from classification.binary.constants import DATASET
from classification.binary.constants import NEGATIVE_LABEL
from classification.binary.constants import POSITIVE_LABEL

import kolena
from kolena.dataset import fetch_dataset
from kolena.dataset import test
from kolena.workflow._datatypes import _get_full_type
from kolena.workflow.annotation import ScoredClassificationLabel

MODELS = ["resnet50v2", "inceptionv3"]
eval_config = {"threshold": 0.5}
id_fields = ["locator"]


def metrics(score, ground_truth_label) -> Dict[str, Any]:
    prediction = ScoredClassificationLabel(label=NEGATIVE_LABEL, score=score)
    if score >= eval_config["threshold"]:
        prediction = ScoredClassificationLabel(label=POSITIVE_LABEL, score=score)

    threshold = eval_config["threshold"]
    if threshold is None:
        threshold = 0.5

    is_positive_prediction = prediction.score >= threshold
    is_positive_sample = ground_truth_label == prediction.label

    return {
        "classification_label": prediction.label,
        "classification_score": prediction.score,
        "threshold": threshold,
        "is_correct": is_positive_prediction == is_positive_sample,
        "is_TP": is_positive_sample and is_positive_prediction,
        "is_FP": not is_positive_sample and is_positive_prediction,
        "is_FN": is_positive_sample and not is_positive_prediction,
        "is_TN": not is_positive_sample and not is_positive_prediction,
    }


def to_kolena_inference(score, multiclass: bool = False) -> List[Dict[str, Any]]:
    # treating this as multiclass classification problem (i.e. both positive and negative labels are
    # equally important) — prediction for both labels are required.

    label = POSITIVE_LABEL
    score = score
    kolena_label = ScoredClassificationLabel(label=label, score=score)
    result = [
        {
            "label": label,
            "score": score,
            "data_type": _get_full_type(kolena_label),
        },
    ]
    if multiclass:
        neg_label = NEGATIVE_LABEL
        neg_score = 1 - score
        neg_kolena_label = ScoredClassificationLabel(label=neg_label, score=neg_score)
        result.append(
            {
                "label": neg_label,
                "score": neg_score,
                "data_type": _get_full_type(neg_kolena_label),
            },
        )
    return result


def upload_results(model_name: str, dataset: str, multiclass: bool) -> None:
    df_results = pd.read_csv(f"s3://{BUCKET}/{DATASET}/results/raw/{model_name}.csv")
    dataset_df = fetch_dataset(dataset)
    df_results = df_results.merge(dataset_df, how="left", on=id_fields)

    raw_inference = df_results["prediction"].apply(lambda x: to_kolena_inference(x, multiclass))
    raw_inference.name = "raw_inference"

    eval_result = df_results.apply(lambda x: pd.Series(metrics(x.prediction, x.label)), axis=1)
    eval_result.columns = [
        "classification_label",
        "classification_score",
        "threshold",
        "is_correct",
        "is_TP",
        "is_FP",
        "is_FN",
        "is_TN",
    ]

    df_results = pd.concat([df_results[["locator"]], raw_inference, eval_result], axis=1)
    test(dataset, model_name, [(eval_config, df_results)])


def run(args: Namespace) -> None:
    kolena.initialize(verbose=True)
    for model_name in args.models:
        upload_results(model_name, DATASET, args.multiclass)


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
        "--multiclass",
        action="store_true",
        default=True,
        help="Option to evaluate dogs-vs-cats as multiclass classification",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
