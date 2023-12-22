# Copyright 2021-2023 Kolena Inc.
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
from question_answering.constants import MODELS
from question_answering.constants import TRUTHFULQA

import kolena
from kolena._experimental.dataset import test
from kolena.workflow.io import dataframe_from_csv

DATASET_TO_METRICS_RESULTS = {
    TRUTHFULQA: {model: f"s3://{BUCKET}/TruthfulQA/results/v1/{model}_with_metrics.csv" for model in MODELS},
    HALUEVALQA: {model: f"s3://{BUCKET}/HaLuEval/evaluation/v1/qa_{model}_with_metrics.csv" for model in MODELS},
}


def main(args: Namespace) -> None:
    kolena.initialize(verbose=True)
    for dataset in args.datasets:
        print(f"Loading {dataset}...")
        for model in args.models:
            print(f"Loading {model} results on {dataset}...")
            df_results = dataframe_from_csv(DATASET_TO_METRICS_RESULTS[dataset][model])
            df_results["gpt4_hallucination_flag"] = df_results["gpt4_hallucination_flag"].replace({True: 1, False: 0})

            test(dataset, model, df_results)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=DATASET_TO_METRICS_RESULTS.keys(),
        choices=DATASET_TO_METRICS_RESULTS.keys(),
        help="Name(s) of the dataset(s) to test.",
    )

    ap.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        choices=MODELS,
        help="Name(s) of the dataset(s) to test.",
    )
    main(ap.parse_args())
