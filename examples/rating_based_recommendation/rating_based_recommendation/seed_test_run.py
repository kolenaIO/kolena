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
import sys
from argparse import ArgumentParser
from argparse import Namespace
from typing import List

import pandas as pd
from rating_based_recommendation.evaluator import compute_per_sample
from rating_based_recommendation.evaluator import evaluate_recommender
from rating_based_recommendation.workflow import Inference
from rating_based_recommendation.workflow import Model
from rating_based_recommendation.workflow import TestCase
from rating_based_recommendation.workflow import TestSample
from rating_based_recommendation.workflow import TestSuite
from rating_based_recommendation.workflow import ThresholdConfiguration

import kolena
from kolena.workflow import test

BUCKET = "kolena-public-datasets"
DATASET = "movielens"


def seed_test_run(model_name: str, test_suite_names: List[str]) -> None:
    df_results = pd.read_csv(f"s3://{BUCKET}/{DATASET}/results/raw/predictions_{model_name}.sample.csv")

    def infer(test_sample: TestSample) -> Inference:
        row = df_results[
            (df_results["userId"] == int(test_sample.user_id.text)) & (df_results["movieId"] == test_sample.movie_id)
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

        inferences = model.load_inferences(TestCase(f"complete :: {DATASET}-10k"))
        results = []
        for ts, gt, inf in inferences:
            metrics = compute_per_sample(gt, inf, ThresholdConfiguration(rating_threshold=3.5))
            metrics = metrics.__dict__
            del metrics["__pydantic_initialised__"]
            metrics["abs_Δ_rating"] = abs(metrics["Δ_rating"])
            results.append(
                dict(
                    user_id=ts.user_id.text,
                    movie_id=ts.movie_id,
                    rating=inf.pred_rating,
                    **metrics,
                ),
            )

        df_results = pd.DataFrame.from_records(results)
        df_results.to_csv(f"s3://kolena-public-datasets/movielens/results/{model_name}_T3_5.csv", index=False)


def main(args: Namespace) -> int:
    kolena.initialize(verbose=True)
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
        default=[
            "movielens-10k",
        ],
        help="Name(s) of test suite(s) to test.",
    )
    sys.exit(main(ap.parse_args()))
