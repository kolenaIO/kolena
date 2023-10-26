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
import os
import sys
from argparse import ArgumentParser
from argparse import Namespace
from typing import List

import pandas as pd
from rating_based_recommendation.evaluator import evaluate_recommender
from rating_based_recommendation.workflow import ThresholdConfiguration
from rating_based_recommendation.workflow import Inference
from rating_based_recommendation.workflow import Model
from rating_based_recommendation.workflow import TestSample
from rating_based_recommendation.workflow import TestSuite

import kolena
from kolena.workflow import test

BUCKET = "kolena-public-datasets"
DATASET = "movielens"


def seed_test_run(model_name: str, test_suite_names: List[str]) -> None:
    df_results = pd.read_csv(f"s3://{BUCKET}/{DATASET}/results/predictions_{model_name}.csv")

    def infer(test_sample: TestSample) -> Inference:
        row = df_results[
            (df_results["userId"] == test_sample.user_id) & (df_results["movieId"] == test_sample.movie_id)
        ]
        rating = 0 if row.empty else row.prediction.values[0]
        return Inference(pred_rating=round(rating, 2))

    model_descriptor = f"{model_name} [{DATASET}-10k]"
    model_metadata = dict(library="lenskit", dataset="movielens-1M")

    model = Model(name=model_descriptor, infer=infer, metadata=model_metadata)

    print(f"Model: {model}")

    configurations = [ThresholdConfiguration(rating_threshold=3.0), ThresholdConfiguration(rating_threshold=3.5)]

    for test_suite_name in test_suite_names:
        test_suite = TestSuite.load(test_suite_name)
        print(f"Test Suite: {test_suite}")
        test(model, test_suite, evaluate_recommender, configurations, reset=True)


def main(args: Namespace) -> int:
    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)
    for model_name in args.models:
        seed_test_run(model_name, args.test_suites)
    return 0


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--models",
        default=["MF_als", "knn"],
        help="Name(s) of model(s) in directory to test",
    )
    ap.add_argument(
        "--test_suites",
        default=[f"{DATASET}-10k :: genre", f"{DATASET}-10k :: age", f"{DATASET}-10k :: occupation"],
        help="Name(s) of test suite(s) to test.",
    )
    sys.exit(main(ap.parse_args()))
