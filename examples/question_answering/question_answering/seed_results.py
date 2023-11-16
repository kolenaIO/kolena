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
from argparse import ArgumentParser
from argparse import Namespace

import pandas as pd
from question_answering.evaluator import compute_metrics
from question_answering.seed_dataset import BUCKET
from question_answering.seed_dataset import DATASET
from question_answering.utils import normalize_string

import kolena
from kolena._experimental.dataset import test
from kolena.workflow.annotation import ClassificationLabel

MODELS = ["gpt-3.5-turbo-0301", "gpt-3.5-turbo", "gpt-4-0314", "gpt-4"]


def main(args: Namespace) -> None:
    kolena.initialize(verbose=True)

    model_file_path = f"s3://{BUCKET}/{DATASET}/results/{args.model}.csv"
    data_file_path = f"s3://{BUCKET}/{DATASET}/metadata/metadata.csv"
    df_data = pd.read_csv(data_file_path)
    df = pd.read_csv(model_file_path)

    threshold = 0.5
    df["clean_answer"] = df["answer"].apply(normalize_string)
    df["answer"] = df["answer"].apply(lambda x: ClassificationLabel(label=x))
    results = df_data[["data_id", "turn", "metadata_answer"]].merge(df, on=["data_id", "turn"])
    df_metrics = results.apply(
        lambda x: compute_metrics(normalize_string(x["metadata_answer"]), x["clean_answer"]),
        result_type="expand",
        axis=1,
    )
    results = pd.concat([results, df_metrics], axis=1)
    results["MEAN_METRIC"] = round((results["BERT_f1"] + results["ROUGE_1"]) / 2, 3)
    results["is_correct"] = results["MEAN_METRIC"] >= threshold
    test(args.dataset, args.model, [(dict(threshold=threshold), results)], on=["data_id", "turn"])


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--dataset", default=DATASET, help="Name of the dataset to test.")
    ap.add_argument("--model", default="gpt-4", choices=MODELS, help="Name of the model to test.")

    main(ap.parse_args())
