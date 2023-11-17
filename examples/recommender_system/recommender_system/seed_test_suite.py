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
import time
from argparse import ArgumentParser
from argparse import Namespace

import pandas as pd
from recommender_system.utils import AGE_STRATIFICATION
from recommender_system.utils import GENRES
from recommender_system.utils import OCCUPATION_STRATIFICATION
from recommender_system.utils import process_metadata
from recommender_system.workflow import GroundTruth
from recommender_system.workflow import Movie
from recommender_system.workflow import TestCase
from recommender_system.workflow import TestSample
from recommender_system.workflow import TestSuite

import kolena

# from recommender_system.utils import ID_AGE_MAP
# from recommender_system.utils import ID_OCCUPATION_MAP

BUCKET = "kolena-public-datasets"
DATASET = "movielens"


def main(args: Namespace) -> int:
    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)

    t0 = time.time()

    df_ratings = pd.read_csv(args.ratings_csv)
    df_movies = pd.read_csv(args.movies_csv)
    df_users = pd.read_csv(args.users_csv)

    movie_metadata, id_title_map = {}, {}
    non_metadata_fields = {"movieId", "title"}
    for record in df_movies.itertuples(index=False):
        fields = set(record._fields)
        movie_metadata[record.movieId] = {f: process_metadata(record, f) for f in fields - non_metadata_fields}
        id_title_map[record.movieId] = record.title

    metadata_by_user_id = {}
    non_metadata_fields = {"userId"}
    for record in df_users.itertuples(index=False):
        fields = set(record._fields)
        metadata_by_user_id[record.userId] = {f: process_metadata(record, f) for f in fields - non_metadata_fields}

    test_samples_and_ground_truths = []
    unique_users = set(df_ratings["userId"])
    for uid in unique_users:
        samples = df_ratings[df_ratings["userId"] == uid].itertuples(index=False)
        test_samples_and_ground_truths.append(
            (
                TestSample(
                    user_id=uid,
                    metadata=metadata_by_user_id[uid],
                ),
                GroundTruth(
                    rated_movies=[
                        Movie(
                            label=id_title_map[sample.movieId],
                            id=sample.movieId,
                            score=round(sample.rating, 2),
                            metadata=movie_metadata[sample.movieId],
                        )
                        for sample in samples
                    ],
                ),
            ),
        )

    t1 = time.time()
    complete_test_case = TestCase(
        name=f"complete :: {DATASET}",
        description=f"All images in {DATASET} dataset",
        test_samples=test_samples_and_ground_truths,
        reset=True,
    )
    print(f"created baseline test case '{complete_test_case.name}' in {time.time() - t1:0.3f} seconds")

    # Metadata Test Cases
    cateogry_subsets = dict(
        genre=GENRES,
        age=AGE_STRATIFICATION.values(),
        occupation=OCCUPATION_STRATIFICATION.values(),
        gender=["M", "F"],
    )

    age_ts_gt_splits = {item: [] for item in cateogry_subsets["age"]}
    occupation_ts_gt_splits = {item: [] for item in cateogry_subsets["occupation"]}
    gender_ts_gt_splits = {item: [] for item in cateogry_subsets["gender"]}

    for ts, gt in test_samples_and_ground_truths:
        age_ts_gt_splits[AGE_STRATIFICATION[ts.metadata["age"]]].append((ts, gt))
        occupation_ts_gt_splits[OCCUPATION_STRATIFICATION[ts.metadata["occupation"]]].append((ts, gt))
        gender_ts_gt_splits[ts.metadata["gender"]].append((ts, gt))

    test_suites = dict(
        age=age_ts_gt_splits,
        occupation=occupation_ts_gt_splits,
        gender=gender_ts_gt_splits,
    )

    for cat, ts in test_suites.items():
        t2 = time.time()
        test_cases = TestCase.init_many(
            [(f"{cat} :: {type} :: {DATASET}", test_samples) for type, test_samples in ts.items()],
            reset=True,
        )
        print(f"created test case genre stratifications in {time.time() - t2:0.3f} seconds")

        test_suite = TestSuite(
            name=f"{DATASET} :: {cat}",
            test_cases=[complete_test_case, *test_cases],
            reset=True,
        )
        print(f"created test suite '{test_suite}' in {time.time() - t2:0.3f} seconds")

    print(f"completed test suite seeding in {time.time() - t0:0.3f} seconds")


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--ratings_csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/meta/ratings.sample.csv",
    )
    ap.add_argument(
        "--movies_csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/meta/movies.csv",
    )
    ap.add_argument(
        "--users_csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/meta/users.csv",
    )
    sys.exit(main(ap.parse_args()))
