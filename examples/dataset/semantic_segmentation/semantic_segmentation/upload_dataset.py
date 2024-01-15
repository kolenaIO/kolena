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
from semantic_segmentation.constants import BUCKET
from semantic_segmentation.constants import DATASET

import kolena
from kolena.annotation import SegmentationMask
from kolena.dataset import upload_dataset
from kolena.io import dataframe_to_csv


def run(args: Namespace) -> None:
    kolena.initialize(verbose=True)
    df_dataset = pd.read_csv(args.dataset_csv, storage_options={"anon": True})
    df_dataset["mask"] = df_dataset["mask"].apply(lambda mask: SegmentationMask(locator=mask, labels={1: "PERSON"}))
    dataframe_to_csv(df_dataset, f"s3://{BUCKET}/{DATASET}/{DATASET}-person.csv", index=False)
    upload_dataset(args.dataset_name, df_dataset)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset-csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/raw/coco-stuff-10k-person.csv",
        help="CSV file specifying dataset. See default CSV for details",
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
