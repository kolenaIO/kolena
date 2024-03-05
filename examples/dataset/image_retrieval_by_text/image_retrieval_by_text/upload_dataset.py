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
from image_retrieval_by_text.constants import BUCKET
from image_retrieval_by_text.constants import DATASET
from image_retrieval_by_text.constants import ID_FIELDS

import kolena
from kolena.asset import ImageAsset
from kolena.dataset import upload_dataset


def transform_data(df_csv: pd.DataFrame) -> pd.DataFrame:
    df_csv["image_url"] = df_csv["image_url"].apply(lambda x: ImageAsset(locator=x))
    # You could add more meaningful metadata here.
    df_csv["aspect_ratio"] = df_csv["width"] / df_csv["height"]
    return df_csv


def run(args: Namespace) -> None:
    kolena.initialize(verbose=True)
    df_csv = pd.read_csv(
        f"s3://{BUCKET}/coco-2014-val/image-retrieval-by-text/dataset/raw/" f"image-retrieval-by-text-raw.csv",
        storage_options={"anon": True},
    )
    df_csv.drop("aspect_ratio", axis=1, inplace=True)
    df_with_metadata = transform_data(df_csv)
    upload_dataset(args.dataset, df_with_metadata, id_fields=ID_FIELDS)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--dataset", type=str, default=DATASET, help="Optionally specify a custom dataset name to upload.")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
