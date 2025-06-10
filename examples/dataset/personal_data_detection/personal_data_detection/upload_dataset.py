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

import pandas as pd
from personal_data_detection.constants import DATA_FILEPATH
from personal_data_detection.constants import DATASET
from personal_data_detection.utils import detect_pii_in_dataframe

from kolena.dataset import upload_dataset


def run(args: Namespace) -> None:
    df = pd.read_csv(DATA_FILEPATH)
    if not detect_pii_in_dataframe(df, allowed_pii_types=args.allowed_pii_types):
        upload_dataset(args.dataset, df)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        default=DATASET,
        help="Optionally specify a custom name for the dataset.",
    )
    ap.add_argument(
        "--allowed_pii_types",
        nargs="+",
        default=[],
        help="Types of PII data to allow in the upload.",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
