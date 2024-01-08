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
from classification.binary.constants import BUCKET
from classification.binary.constants import DATASET
from classification.binary.constants import ID_FIELDS

import kolena
from kolena.annotation import ClassificationLabel
from kolena.dataset import upload_dataset


def run(args: Namespace) -> None:
    df = pd.read_csv(f"s3://{BUCKET}/{DATASET}/raw/{DATASET}.csv", storage_options={"anon": True})
    df["label"] = df["label"].apply(lambda label: ClassificationLabel(label))
    kolena.initialize(verbose=True)
    upload_dataset(args.dataset, df, ID_FIELDS)


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
