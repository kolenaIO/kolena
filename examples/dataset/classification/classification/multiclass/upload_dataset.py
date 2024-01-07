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
from classification.multiclass.constants import BUCKET
from classification.multiclass.constants import DATASET
from classification.multiclass.constants import ID_FIELDS

import kolena
from kolena.dataset import register_dataset
from kolena.workflow.annotation import ClassificationLabel


def run(args: Namespace) -> None:
    df = pd.read_csv(f"s3://{BUCKET}/{DATASET}/raw/{DATASET}.csv", storage_options={"anon": True})
    df["ground_truth"] = df["ground_truth"].apply(lambda label: ClassificationLabel(label))
    kolena.initialize(verbose=True)
    register_dataset(args.dataset, df, ID_FIELDS)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset",
        default=DATASET,
        help=f"Custom name for the {DATASET} dataset to upload.",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()