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
import sys
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
from kolena._experimental.dataset._dataset import _to_deserialized_dataframe
from kolena._experimental.dataset._evaluation import _fetch_dataset
from kolena._experimental.dataset._evaluation import test
from kolena._experimental.dataset.common import COL_DATAPOINT
from kolena.workflow._datatypes import _get_full_type
from kolena.workflow.annotation import ScoredClassificationLabel

eval_config = {"threshold": 0.5}
id_fields = ["locator"]


def to_kolena_result(score, ground_truth_label) -> Dict[str, Any]:
    label = NEGATIVE_LABEL
    correct = False
    if score >= eval_config["threshold"]:
        label = POSITIVE_LABEL
    if ground_truth_label.label == label:
        correct = True
    return {
        "label": label,
        "correct": correct,
        "score": score,
        "data_type": _get_full_type(ScoredClassificationLabel(label=label, score=score)),
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
    df_results = pd.read_csv(
        f"s3://{BUCKET}/{DATASET}/results/predictions_{model_name}.csv",
        storage_options={"anon": True},
    )
    dataset_df = _fetch_dataset(dataset)
    dataset_df = _to_deserialized_dataframe(dataset_df, column=COL_DATAPOINT)
    df_results = df_results.merge(
        dataset_df,
        how="left",
        on=id_fields,
    )
    raw_inference = df_results["prediction"].apply(lambda x: to_kolena_inference(x, multiclass))
    raw_inference.name = "raw_inference"
    eval_result = df_results.apply(lambda x: to_kolena_result(x.prediction, x.label), axis=1)
    eval_result.name = "prediction"
    df_results = pd.concat([df_results[["locator"]], raw_inference, eval_result], axis=1)
    test(dataset, model_name, [(eval_config, df_results)])


# TODO: Add logging to the link to result page
def main(args: Namespace) -> int:
    kolena.initialize(verbose=True)
    for model_name in args.models:
        upload_results(model_name, args.dataset, args.multiclass)
    return 0


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--models",
        default=["resnet50v2", "inceptionv3"],
        nargs="+",
        help="Name(s) of model(s) in directory to test",
    )
    ap.add_argument(
        "--dataset",
        default="dogs-vs-cats",
        nargs="+",
        help="Name of dataset to test.",
    )
    ap.add_argument(
        "--multiclass",
        action="store_true",
        default=True,
        help="Option to evaluate dogs-vs-cats as multiclass classification",
    )
    sys.exit(main(ap.parse_args()))
