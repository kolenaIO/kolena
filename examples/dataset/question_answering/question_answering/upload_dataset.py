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

from question_answering.constants import DATASET_TO_LOCATOR

import kolena
from kolena.dataset import upload_dataset
from kolena.workflow.io import dataframe_from_csv


def run(args: Namespace) -> None:
    kolena.initialize(verbose=True)
    for dataset in args.datasets:
        print(f"loading {dataset}...")
        df_datapoint = dataframe_from_csv(DATASET_TO_LOCATOR[dataset])
        upload_dataset(dataset, df_datapoint, id_fields=["id"])


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=DATASET_TO_LOCATOR.keys(),
        choices=DATASET_TO_LOCATOR.keys(),
        help="Name(s) of the dataset(s) to register.",
    )

    run(ap.parse_args())


if __name__ == "__main__":
    main()
