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
from text_summarization.constants import BUCKET
from text_summarization.constants import DATASET
from text_summarization.constants import ID_FIELD
from text_summarization.constants import MODELS
from text_summarization.metrics import compute_metrics
from tqdm import tqdm

import kolena
from kolena.dataset import fetch_dataset
from kolena.dataset import upload_results


def run(args: Namespace) -> int:
    kolena.initialize(verbose=True)
    for model in args.models:
        df_dataset = fetch_dataset(args.dataset)
        df_inferences = pd.read_csv(f"s3://{BUCKET}/{DATASET}/results/raw/{model}.csv")
        df = df_inferences.merge(df_dataset, on=ID_FIELD)

        results = []
        for record in tqdm(df.itertuples(index=False), total=len(df)):
            metrics = compute_metrics(record.text_summary, record.inference)
            results.append(
                dict(
                    **record._asdict(),
                    **metrics,
                ),
            )

        df_results = pd.DataFrame.from_records(results)
        upload_results(args.dataset, model, df_results)
    return 0


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS, help="Name(s) of model(s) to test.")
    ap.add_argument("--dataset", default=DATASET, help="Name of the dataset to test.")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
