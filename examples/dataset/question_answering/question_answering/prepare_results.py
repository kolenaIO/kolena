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
from question_answering.closed_domain_metrics import compute_closed_domain_metrics
from question_answering.constants import DATASET_TO_INFERENCES
from question_answering.constants import DATASET_TO_LOCATOR
from question_answering.constants import DATASET_TO_RESULTS
from question_answering.constants import HALUEVALQA
from question_answering.constants import MODELS
from question_answering.open_domain_metrics import compute_open_domain_metrics
from tqdm import tqdm


def compute_open_domain_metrics_for_dataset(df_datapoints: pd.DataFrame, df_results: pd.DataFrame) -> pd.DataFrame:
    df_merged = df_datapoints.merge(df_results, on=["id"])

    dataset_metrics = []
    for record in tqdm(df_merged.itertuples(), total=len(df_merged)):
        answers = [getattr(record, f"answer_{i}") for i in range(5)]
        datapoint_metrics = compute_open_domain_metrics(record.question, record.best_answer, record.answer_0, answers)
        datapoint_metrics["id"] = record.id
        dataset_metrics.append(datapoint_metrics)

    return pd.DataFrame(dataset_metrics)


def compute_closed_domain_metrics_for_dataset(df_datapoints: pd.DataFrame, df_results: pd.DataFrame) -> pd.DataFrame:
    df_merged = df_datapoints.merge(df_results, on=["id"])

    dataset_metrics = []
    for record in tqdm(df_merged[:1000].itertuples(), total=len(df_merged)):
        answers = [getattr(record, f"answer_{i}") for i in range(5)]
        datapoint_metrics = compute_closed_domain_metrics(
            record.text,
            record.question,
            record.right_answer,
            record.answer_0,
            answers,
        )
        datapoint_metrics["id"] = record.id
        dataset_metrics.append(datapoint_metrics)

    return pd.DataFrame(dataset_metrics)


def main(args: Namespace) -> None:
    for dataset in args.datasets:
        print(f"Loading {dataset}...")
        df_datapoints = pd.read_csv(DATASET_TO_LOCATOR[dataset], storage_options={"anon": True})
        for model in args.models:
            print(f"Loading {model} results on {dataset}...")
            df_results = pd.read_csv(DATASET_TO_INFERENCES[dataset][model], storage_options={"anon": True})
            if dataset == HALUEVALQA:
                df_metrics = compute_closed_domain_metrics_for_dataset(df_datapoints, df_results)
            else:
                df_metrics = compute_open_domain_metrics_for_dataset(df_datapoints, df_results)
            df_metrics_with_results = df_results.merge(df_metrics, on=["id"])
            df_metrics_with_results.to_csv(DATASET_TO_RESULTS[dataset][model], index=False)
            print(f"Saved the results wtih metrics to {DATASET_TO_RESULTS[dataset][model]}...")


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=DATASET_TO_LOCATOR.keys(),
        choices=DATASET_TO_LOCATOR.keys(),
        help="Name(s) of the dataset(s) to prepare results for.",
    )

    ap.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        choices=MODELS,
        help="Name(s) of the dataset(s) to prepare results for.",
    )
    main(ap.parse_args())
