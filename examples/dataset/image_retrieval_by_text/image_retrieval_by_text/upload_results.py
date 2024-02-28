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
import json
from argparse import ArgumentParser
from argparse import Namespace
from dataclasses import dataclass
from typing import List

import pandas as pd
from image_retrieval_by_text.constants import BUCKET
from image_retrieval_by_text.constants import DATASET
from image_retrieval_by_text.constants import MODELS

import kolena
from kolena.asset import ImageAsset
from kolena.dataset import upload_results


@dataclass(frozen=True)
class ImageAssetWithScoreAndRank(ImageAsset):
    similarity: float
    rank: int


def _transform_top_10(top_ten: list[dict]) -> List[ImageAssetWithScoreAndRank]:
    current_rank = 1
    new_top_ten = []
    for item in top_ten:
        new_top_ten.append(
            ImageAssetWithScoreAndRank(
                similarity=item["similarity"],
                locator=item["locator"],
                rank=current_rank,
            ),
        )
        current_rank += 1
    return new_top_ten


def transform_data(df_raw_csv: pd.DataFrame) -> pd.DataFrame:
    df_raw_csv["top_10"] = df_raw_csv["top_10"].apply(
        lambda x: _transform_top_10(json.loads(x)),
    )
    return df_raw_csv


def run(args: Namespace) -> None:
    kolena.initialize(verbose=True)

    pred_df_csv = pd.read_csv(
        f"s3://{BUCKET}/image-retrieval-by-text/results/raw/{args.model}-raw.csv",
        storage_options={"anon": True},
    )
    pred_df = transform_data(pred_df_csv)

    upload_results(args.dataset, args.model, pred_df)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--model", type=str, choices=MODELS, default=MODELS[0], help="Name of the model to test.")
    ap.add_argument("--dataset", type=str, default=DATASET, help="Optionally specify a custom dataset name to test.")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
