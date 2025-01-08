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
from retrieval_augmented_generation.constants import DATASET
from retrieval_augmented_generation.constants import MODEL_NAME
from retrieval_augmented_generation.constants import S3_BUCKET
from retrieval_augmented_generation.utils import to_locator

from kolena.asset import DocumentAsset
from kolena.dataset import upload_results


def to_documents(retrieved_contents: list[dict[str, str]]) -> list:
    if not retrieved_contents:
        return []

    documents = []
    for content in retrieved_contents:
        documents.append(
            DocumentAsset(
                locator=to_locator(content["doc_name"]),
                content=content["content"],  # type: ignore[call-arg]
                page_number=content["page_number"],  # type: ignore[call-arg]
            ),
        )

    return documents


def run(args: Namespace) -> None:
    model_name = MODEL_NAME[args.model]
    df_results = pd.read_json(f"{S3_BUCKET}/{DATASET}/results/raw/{model_name}.jsonl", lines=True)
    df_results["retrieved_contents"] = df_results["retrieved_contents"].apply(to_documents)
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
    run(ap.parse_args())


if __name__ == "__main__":
    main()
