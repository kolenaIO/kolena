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
from recommender_system.evaluator import evaluate_recommender
from recommender_system.utils import process_metadata
from recommender_system.workflow import Inference
from recommender_system.workflow import Model
from recommender_system.workflow import Movie
from recommender_system.workflow import RecommenderConfiguration
from recommender_system.workflow import TestSample
from recommender_system.workflow import TestSuite

import kolena
from kolena.workflow import test

BUCKET = "kolena-public-datasets"
DATASET = "movielens"


def seed_test_run(model_name: str, test_suites: List[str], movies_csv: str) -> None:
    # df_results = pd.read_csv(f"s3://{BUCKET}/{DATASET}/results/predictions_{model_name}.sample.csv")
    df_results = pd.read_csv(f"predictions_{model_name}.csv")
    df_movies = pd.read_csv(movies_csv)

    movie_metadata, id_title_map = {}, {}
    non_metadata_fields = {"movieId", "title"}
    for record in df_movies.itertuples(index=False):
        fields = set(record._fields)
        movie_metadata[record.movieId] = {f: process_metadata(record, f) for f in fields - non_metadata_fields}
        id_title_map[record.movieId] = record.title

    def infer(test_sample: TestSample) -> Inference:
        user_recommendations = df_results[df_results["userId"] == test_sample.user_id]
        sorted_recs = user_recommendations.sort_values(by=["prediction"], ascending=False)

        return Inference(
            recommendations=[
                Movie(
                    label=id_title_map[sample.movieId],
                    score=sample.prediction,
                    id=sample.movieId,
                    metadata=movie_metadata[sample.movieId],
                )
                for sample in sorted_recs.itertuples(index=False)
            ],
        )

    model_descriptor = f"{model_name} [{DATASET}]"
    model_metadata = dict(library="lenskit", dataset="movielens-1M")

    model = Model(name=model_descriptor, infer=infer, metadata=model_metadata)

    configurations = [RecommenderConfiguration(k=10)]

    for test_suite_name in test_suites:
        test_suite = TestSuite.load(test_suite_name)
        test(model, test_suite, evaluate_recommender, configurations, reset=False)


def main(args: Namespace) -> int:
    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)
    for model_name in args.models:
        seed_test_run(model_name, args.test_suites, args.movies_csv)
    return 0


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--models",
        default=["dummy"],
        help="Name(s) of model(s) in directory to test",
    )
    ap.add_argument(
        "--test_suites",
        default=[
            f"{DATASET} :: age",
            # f"{DATASET} :: occupation",
            # f"{DATASET} :: gender",
        ],
        help="Name(s) of test suite(s) to test.",
    )
    ap.add_argument(
        "--movies_csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/meta/movies.csv",
    )
    sys.exit(main(ap.parse_args()))
