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
from typing import Optional

import pandas as pd
from retrieval_augmented_generation.constants import DATASET
from retrieval_augmented_generation.constants import ID_FIELDS
from retrieval_augmented_generation.constants import S3_BUCKET

from kolena.asset import DocumentAsset
from kolena.dataset import upload_dataset


def to_locator(filename: str) -> str:
    return f"{S3_BUCKET}/{DATASET}/data/{filename}.pdf"


def to_document(evidence: list[dict[str, str]]) -> Optional[DocumentAsset]:
    if len(evidence) > 0:
        return DocumentAsset(to_locator(evidence[0]["doc_name"]))

    return None


def get_pages(evidence: list[dict[str, str]]) -> str:
    pages = [e["evidence_page_num"] for e in evidence]
    return ", ".join(map(str, pages))


def run(args: Namespace) -> None:
    df_dataset = pd.read_json(args.dataset_jsonl, lines=True)
    df_dataset["document"] = df_dataset["evidence"].apply(to_document)
    df_dataset["relevant_pages"] = df_dataset["evidence"].apply(get_pages)
    upload_dataset(args.dataset_name, df_dataset, id_fields=ID_FIELDS)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset-jsonl",
        type=str,
        default=f"{S3_BUCKET}/{DATASET}/raw/financebench_open_source.jsonl",
        help="JSONL file specifying dataset. See default JSONL for details",
    )
    ap.add_argument(
        "--dataset-name",
        type=str,
        default=DATASET,
        help="Optionally specify a name of the dataset",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
