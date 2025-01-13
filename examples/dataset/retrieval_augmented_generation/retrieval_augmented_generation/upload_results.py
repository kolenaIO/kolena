# Copyright 2021-2025 Kolena Inc.
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

import pandas as pd
from retrieval_augmented_generation.constants import DATASET
from retrieval_augmented_generation.constants import MODEL_NAME
from retrieval_augmented_generation.constants import S3_BUCKET
from retrieval_augmented_generation.metrics import compute_metrics

from kolena.asset import DocumentAsset
from kolena.dataset import download_dataset
from kolena.dataset import upload_results


def to_documents(retrieved_contents: list[dict[str, Any]]) -> list:
    if not retrieved_contents:
        return []

    documents = []
    for doc in retrieved_contents:
        documents.append(
            DocumentAsset(
                locator=doc["locator"],
                content=doc["content"],  # type: ignore[call-arg]
                page_number=doc["page_number"],  # type: ignore[call-arg]
            ),
        )

    return documents


def run(args: Namespace) -> None:
    model_name = MODEL_NAME[args.model]
    df_results = pd.read_json(f"{S3_BUCKET}/{DATASET}/results/raw/{model_name}.jsonl", lines=True)
    df_results["retrieved_contents"] = df_results["retrieved_contents"].apply(to_documents)
    if args.evaluate:
        df_dataset = download_dataset(args.dataset_name, include_extracted_properties=True)
        df_metrics = compute_metrics(df_dataset, df_results)
        df_results = pd.concat([df_results, df_metrics], axis=1)
    upload_results(args.dataset_name, model_name, df_results)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument(
        "model",
        type=str,
        default="baseline",
        nargs="?",
        choices=list(MODEL_NAME.keys()),
        help="Name of the model to test.",
    )
    ap.add_argument(
        "--dataset-name",
        type=str,
        default=DATASET,
        help="Optionally specify a custom dataset name to test.",
    )
    ap.add_argument(
        "--evaluate",
        action="store_true",
        help="Computes metrics on the model results. Requires dataset with ground truth.",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
