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
from rain_forecast.constants import BUCKET
from rain_forecast.constants import DATASET
from rain_forecast.constants import EVAL_CONFIG
from rain_forecast.constants import MODEL_NAME
from rain_forecast.metrics import compute_metrics
from tqdm import tqdm

import kolena
from kolena.dataset import test


def run(args: Namespace) -> int:
    kolena.initialize(verbose=True)
    df = pd.read_csv(f"s3://{BUCKET}/{DATASET}/results/raw/{args.model}.csv")

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
    test(args.dataset, MODEL_NAME[args.model], [(EVAL_CONFIG, df_results)])
    return 0


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--model", type=str, default="ann", choices=MODEL_NAME.keys(), help="Name of model to test.")
    ap.add_argument("--dataset", default=DATASET, help="Name of dataset to use for testing.")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
