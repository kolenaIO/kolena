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
from rating_based_recommendation.workflow import GroundTruth
from rating_based_recommendation.workflow import TestCase
from rating_based_recommendation.workflow import TestSample
from rating_based_recommendation.workflow import TestSuite
from rating_based_recommendation.utils import (
    ID_OCCUPATION_MAP,
    ID_AGE_MAP,
    OCCUPATION_STRATIFICATION,
    AGE_STRATIFICATION,
)

import kolena

BUCKET = "kolena-public-datasets"
DATASET = "movielens"


def main(args: Namespace) -> int:
    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)

    t0 = time.time()

    df_ratings = pd.read_csv(args.ratings_csv)
    df_movies = pd.read_csv(args.movies_csv)
    df_users = pd.read_csv(args.users_csv)

    def process_metadata(record, f):
        value = getattr(record, f)
        if f == "genres":
            return value.split("|")
        elif f == "age":
            return ID_AGE_MAP[value]
        elif f == "occupation":
            return ID_OCCUPATION_MAP[value]

        return value

    metadata_by_movie_id = {}
    for record in df_movies.itertuples(index=False):
        fields = set(record._fields)
        fields.remove("movieId")
        metadata_by_movie_id[record.movieId] = {f: process_metadata(record, f) for f in fields}

    metadata_by_user_id = {}
    for record in df_users.itertuples(index=False):
        fields = set(record._fields)
        fields.remove("userId")
        metadata_by_user_id[record.userId] = {f: process_metadata(record, f) for f in fields}

    test_samples_and_ground_truths = [
        (
            TestSample(
                user_id=record.userId,
                movie_id=record.movieId,
                metadata={**metadata_by_movie_id[record.movieId], **metadata_by_user_id[record.userId]},
            ),
            GroundTruth(real_rating=record.rating),
        )
        for record in df_ratings.itertuples(index=False)
    ]

    print(f"preparing {len(test_samples_and_ground_truths)} test samples")

    t1 = time.time()
    complete_test_case = TestCase(
        name=f"complete :: {DATASET}-10k",
        description=f"All images in {DATASET} dataset",
        test_samples=test_samples_and_ground_truths,
        reset=True,
    )
    print(f"created baseline test case '{complete_test_case.name}' in {time.time() - t1:0.3f} seconds")

    # Metadata Test Cases
    cateogry_subsets = dict(
        genre=[
            "Adventure",
            "Animation",
            "Children's",
            "Comedy",
            "Fantasy",
            "Romance",
            "Drama",
            "Action",
            "Crime",
            "Thriller",
            "Horror",
            "Mystery",
            "Sci-Fi",
            "IMAX",
            "Documentary",
            "War",
            "Musical",
            "Western",
            "Film-Noir",
            "(no genres listed)",
        ],
        age=AGE_STRATIFICATION.values(),
        occupation=OCCUPATION_STRATIFICATION.values(),
        gender=["M", "F"],
    )

    genre_ts_gt_splits = {item: [] for item in cateogry_subsets["genre"]}
    age_ts_gt_splits = {item: [] for item in cateogry_subsets["age"]}
    occupation_ts_gt_splits = {item: [] for item in cateogry_subsets["occupation"]}
    gender_ts_gt_splits = {item: [] for item in cateogry_subsets["gender"]}

    for ts, gt in test_samples_and_ground_truths:
        for genre in ts.metadata["genres"]:
            genre_ts_gt_splits[genre].append((ts, gt))
        age_ts_gt_splits[AGE_STRATIFICATION[ts.metadata["age"]]].append((ts, gt))
        occupation_ts_gt_splits[OCCUPATION_STRATIFICATION[ts.metadata["occupation"]]].append((ts, gt))
        gender_ts_gt_splits[ts.metadata["gender"]].append((ts, gt))

    test_suites = dict(
        genre=genre_ts_gt_splits,
        age=age_ts_gt_splits,
        occupation=occupation_ts_gt_splits,
        gender=gender_ts_gt_splits,
    )

    for cat, ts in test_suites.items():
        t2 = time.time()
        test_cases = TestCase.init_many(
            [(f"{cat} :: {type} :: {DATASET}-10k", test_samples) for type, test_samples in ts.items()],
            reset=True,
        )
        print(f"created test case genre stratifications in {time.time() - t2:0.3f} seconds")

        test_suite = TestSuite(
            name=f"{DATASET}-10k :: {cat}",
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
