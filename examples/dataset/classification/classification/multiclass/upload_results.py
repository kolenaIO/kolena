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
from classification.multiclass.constants import BUCKET
from classification.multiclass.constants import DATASET
from classification.multiclass.constants import ID_FIELDS

from kolena.dataset import download_dataset
from kolena.dataset import upload_results
from kolena.workflow.annotation import ScoredClassificationLabel


MODELS = ["resnet50v2", "inceptionv3"]
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def compute_metrics(ground_truth_label: str, inferences: List[ScoredClassificationLabel]) -> Dict[str, Any]:
    sorted_infs = sorted(inferences, key=lambda x: x.score, reverse=True)
    predicted_match = sorted_infs[0]
    predicted_label, predicted_score = predicted_match.label, predicted_match.score

    return dict(
        classification=ScoredClassificationLabel(label=predicted_label, score=predicted_score),
        margin=predicted_score - sorted_infs[1].score if len(sorted_infs) >= 2 else None,
        is_correct=predicted_label == ground_truth_label,
    )


def list_inferences(scores: List[float]) -> List[ScoredClassificationLabel]:
    return [ScoredClassificationLabel(class_name, score) for class_name, score in zip(CLASSES, scores)]


def run(args: Namespace) -> None:
    dataset_df = download_dataset(args.dataset)
    df_results = pd.read_csv(
        f"s3://{BUCKET}/{DATASET}/results/raw/{args.model}.csv",
        storage_options={"anon": True},
    )

    df_results = df_results.merge(dataset_df, how="left", on=ID_FIELDS)
    df_results["inferences"] = df_results[CLASSES].apply(lambda row: list_inferences(row.tolist()), axis=1)
    eval_result = df_results.apply(
        lambda row: pd.Series(compute_metrics(row.ground_truth.label, row.inferences)),
        axis=1,
    )
    df_results = pd.concat([df_results[ID_FIELDS], df_results["inferences"], eval_result], axis=1)

    upload_results(args.dataset, args.model, df_results)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("model", type=str, choices=MODELS, help="Name of the model to test.")
    ap.add_argument("--dataset", type=str, default=DATASET, help="Optionally specify a custom dataset name to test.")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
