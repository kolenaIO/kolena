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

from person_detection.constants import BUCKET
from person_detection.constants import DATASET_DIR
from person_detection.constants import DATASET_NAME
from person_detection.constants import ID_FIELDS

import kolena.io
from kolena.dataset import upload_dataset


def run(args: Namespace) -> None:
    kolena.initialize(verbose=True)
    df_metadata = kolena.io.dataframe_from_csv(
        f"s3://{BUCKET}/{DATASET_DIR}/{DATASET_NAME}.csv",
        storage_options={"anon": True},
    )
    sample_count = args.sample_count
    if sample_count:
        df_metadata.sort_values(by="locator", inplace=True, ignore_index=True)
        df_metadata = df_metadata[:sample_count]
    upload_dataset(args.dataset, df_metadata, id_fields=ID_FIELDS)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        default=DATASET_NAME,
        help="Optionally specify a custom dataset name to upload.",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
