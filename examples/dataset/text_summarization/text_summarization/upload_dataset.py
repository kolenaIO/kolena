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

from text_summarization.constants import BUCKET
from text_summarization.constants import DATASET
from text_summarization.constants import ID_FIELD

from kolena.dataset import upload_dataset
from kolena.io import dataframe_from_csv


def run(args: Namespace) -> None:
    df_dataset = dataframe_from_csv(args.dataset_csv, storage_options={"anon": True})
    upload_dataset(args.dataset, df_dataset, id_fields=[ID_FIELD])


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset-csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/CNN-DailyMail.csv",
        help="CSV file specifying dataset. See default CSV for details",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default=DATASET,
        help="Optionally specify a name of the dataset to upload.",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
