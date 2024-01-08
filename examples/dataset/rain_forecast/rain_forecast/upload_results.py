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
from typing import Any
from typing import Dict

import pandas as pd
from rain_forecast.constants import BUCKET
from rain_forecast.constants import DATASET
from tqdm import tqdm

import kolena
from kolena.dataset import upload_results


MODEL_NAME = {
    "ann": "ann-batch32-epoch150",
    "logreg": "logreg-liblinear",
}
EVAL_CONFIG = {"threshold": 0.5}


def compute_metrics(ground_truth: str, inference: float, threshold: float = 0.5) -> Dict[str, Any]:
    if ground_truth == "Yes" or ground_truth == "No":
        gt = ground_truth == "Yes"
        inf = inference >= threshold
        metrics = dict(
            missing_ground_truth=False,
            is_correct=gt == inf,
            is_TP=gt == inf and gt,
            is_FP=gt != inf and not gt,
            is_FN=gt != inf and gt,
            is_TN=gt == inf and not gt,
        )
        return metrics

    return dict(missing_ground_truth=True, is_correct=None, is_TP=None, is_FP=None, is_FN=None, is_TN=None)


def run(args: Namespace) -> None:
    df = pd.read_csv(f"s3://{BUCKET}/{DATASET}/results/raw/{args.model}.csv", storage_options={"anon": True})

    results = []
    for record in tqdm(df.itertuples(), total=len(df)):
        metrics = compute_metrics(record.RainTomorrow, record.ChanceOfRain, threshold=EVAL_CONFIG["threshold"])
        results.append(
            dict(
                Date=record.Date,
                Location=record.Location,
                ChanceOfRain=record.ChanceOfRain,
                Threshold=EVAL_CONFIG["threshold"],
                **metrics,
            ),
        )

    df_results = pd.DataFrame.from_records(results)

    kolena.initialize(verbose=True)
    upload_results(args.dataset, MODEL_NAME[args.model], [(EVAL_CONFIG, df_results)])


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("model", type=str, choices=list(MODEL_NAME.keys()), help="Name of the model to test.")
    ap.add_argument(
        "--dataset",
        type=str,
        default=DATASET,
        help="Optionally specify a custom dataset name to test.",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
