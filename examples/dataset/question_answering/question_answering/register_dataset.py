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

from question_answering.constants import BUCKET
from question_answering.constants import HALUEVALQA
from question_answering.constants import TRUTHFULQA

import kolena
from kolena._experimental.dataset import register_dataset
from kolena.workflow.io import dataframe_from_csv

DATASETS = {
    TRUTHFULQA: f"s3://{BUCKET}/TruthfulQA/v1/TruthfulQA_QA.csv",
    HALUEVALQA: f"s3://{BUCKET}/HaLuEval/data/v1/qa_data.csv",
}


def main(args: Namespace) -> None:
    kolena.initialize(verbose=True)
    for dataset in args.datasets:
        print(f"Loading {dataset}...")
        df_datapoint = dataframe_from_csv(DATASETS[dataset])
        register_dataset(dataset, df_datapoint, id_fields=["id"])


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS.keys(),
        choices=DATASETS.keys(),
        help="Name(s) of the dataset(s) to register.",
    )

    main(ap.parse_args())
