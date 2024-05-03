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
from semantic_textual_similarity.constants import BUCKET
from semantic_textual_similarity.constants import DATASET
from semantic_textual_similarity.constants import ID_FIELD
from semantic_textual_similarity.constants import MODELS
from semantic_textual_similarity.metrics import compute_metrics
from tqdm import tqdm

from kolena.dataset import download_dataset
from kolena.dataset import upload_results


def run(args: Namespace) -> None:
    model = MODELS[args.model]
    df_dataset = download_dataset(args.dataset)
    df_inferences = pd.read_csv(f"s3://{BUCKET}/{DATASET}/results/raw/{model}.csv", storage_options={"anon": True})
    df = df_inferences.merge(df_dataset[["id", "similarity"]], on=ID_FIELD)

    results = []
    for record in tqdm(df.itertuples(index=False), total=len(df)):
        metrics = compute_metrics(record.similarity, record.cos_similarity)
        result = record._asdict()
        result.pop("similarity")
        results.append(dict(**result, **metrics))

    df_results = pd.DataFrame.from_records(results)
    upload_results(args.dataset, model, df_results)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("model", type=str, choices=list(MODELS.keys()), help="Name of the model to test.")
    ap.add_argument("--dataset", default=DATASET, help="Optionally specify a custom dataset name to test.")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
