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
from person_detection.constants import DATASET
from person_detection.constants import ID_FIELDS
from person_detection.constants import LICENSE

import kolena.io
from kolena.dataset import upload_dataset


def run(args: Namespace) -> None:
    kolena.initialize(verbose=True)
    df_metadata = kolena.io.dataframe_from_csv(
        f"s3://{BUCKET}/{DATASET}/{DATASET}.csv",
        storage_options={"anon": True},
    )
    if args.license == LICENSE:
        df_metadata = df_metadata[df_metadata.apply(lambda row: "NonCommercial" not in row["license_name"], axis=1)]
    upload_dataset(args.dataset, df_metadata, id_fields=ID_FIELDS)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--dataset", type=str, default=DATASET, help="Optionally specify a custom dataset name to upload.")
    ap.add_argument("--license", type=str, default=LICENSE, help="Optionally specify license type for data to upload")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
