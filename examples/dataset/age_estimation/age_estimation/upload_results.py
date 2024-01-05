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

import kolena
from kolena.dataset import fetch_dataset
from kolena.dataset import test

from age_estimation.constants import BUCKET
from age_estimation.constants import DATASET
from tqdm import tqdm


def run(args: Namespace) -> None:
    model = args.model

    kolena.initialize(verbose=True)
    dataset_df = fetch_dataset(DATASET)
    age_by_locator = {record["locator"]: record["age"]
                      for record in dataset_df.to_dict(orient="records")}

    df_results = pd.read_csv(f"s3://{BUCKET}/{DATASET}/results/raw/{model}.csv")

    results = []
    for record in tqdm(df_results.itertuples(), total=len(df_results)):
        error = age_by_locator[record.locator] - record.age
        results.append({"locator": record.locator,
                        "age": record.age,
                        "fail_to_detect": record.age is None or record.age < 0,
                        "error": error,
                        "absolute_error": abs(error),
                        "square_error": error ** 2})

    df_metrics = pd.DataFrame.from_records(results)
    df_metrics.to_csv(f"{model}.csv")

    test(DATASET, model, df_metrics)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("model", type=str, choices=["ssr", "deepface"], help="Name of model to test.")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
