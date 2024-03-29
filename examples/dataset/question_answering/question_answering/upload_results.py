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

from question_answering.constants import DATASET_TO_RESULTS
from question_answering.constants import MODELS

from kolena.dataset import upload_results
from kolena.workflow.io import dataframe_from_csv


def main(args: Namespace) -> None:
    for dataset in args.datasets:
        print(f"loading {dataset}...")
        for model in args.models:
            print(f"loading {model} results on {dataset}...")
            df_results = dataframe_from_csv(DATASET_TO_RESULTS[dataset][model])
            upload_results(dataset, model, df_results)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=DATASET_TO_RESULTS.keys(),
        choices=DATASET_TO_RESULTS.keys(),
        help="Name(s) of the dataset(s) to test.",
    )

    ap.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        choices=MODELS,
        help="Name(s) of the model(s) to test.",
    )
    main(ap.parse_args())
