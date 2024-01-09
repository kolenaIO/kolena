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
from object_detection_2d.constants import BUCKET
from object_detection_2d.constants import DATASET
from object_detection_2d.constants import ID_FIELDS
from object_detection_2d.data_loader import load_data

import kolena
from kolena.dataset import upload_dataset


def run(args: Namespace) -> None:
    kolena.initialize(verbose=True)
    df_metadata_csv = pd.read_csv(
        f"s3://{BUCKET}/{args.dataset}/meta/metadata_attribution2.0_transportation.csv",
        storage_options={"anon": True},
    )
    df_metadata = load_data(df_metadata_csv[:1000], is_pred=False)
    upload_dataset(args.dataset, df_metadata, id_fields=ID_FIELDS)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        default=DATASET,
        help="Optionally specify a custom dataset name to upload.",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
