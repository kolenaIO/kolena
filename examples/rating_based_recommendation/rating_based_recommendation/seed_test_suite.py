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

import pandas as pd
from rating_based_recommendation.utils import AGE_STRATIFICATION
from rating_based_recommendation.utils import ID_AGE_MAP
from rating_based_recommendation.utils import ID_OCCUPATION_MAP
from rating_based_recommendation.utils import OCCUPATION_STRATIFICATION
from rating_based_recommendation.workflow import GroundTruth
from rating_based_recommendation.workflow import TestCase
from rating_based_recommendation.workflow import TestSample
from rating_based_recommendation.workflow import TestSuite

import kolena
from kolena.workflow import Text
from kolena.workflow.io import dataframe_to_csv

BUCKET = "kolena-public-datasets"
DATASET = "movielens"


def main(args: Namespace) -> int:
    kolena.initialize(verbose=True)

    df_ratings = pd.read_csv(args.ratings_csv)
    df_movies = pd.read_csv(args.movies_csv)
    df_users = pd.read_csv(args.users_csv)

    def process_metadata(record, f):
        value = getattr(record, f)
        if f == "genres":
            return [str(v) for v in value.split("|")]
        elif f == "age":
            return ID_AGE_MAP[value]
        elif f == "occupation":
            return ID_OCCUPATION_MAP[value]

        return value

    metadata_by_movie_id = {}
    movie_id_title = {}
    for record in df_movies.itertuples(index=False):
        fields = set(record._fields)
        fields.remove("movieId")
        fields.remove("title")
        metadata_by_movie_id[record.title] = {f: process_metadata(record, f) for f in fields}
        movie_id_title[record.movieId] = record.title

    metadata_by_user_id = {}
    for record in df_users.itertuples(index=False):
        fields = set(record._fields)
        fields.remove("userId")
        metadata_by_user_id[record.userId] = {f: process_metadata(record, f) for f in fields}

    test_samples_and_ground_truths = [
        (
            TestSample(
                user_id=Text(text=str(record.userId)),
                title=Text(text=movie_id_title[record.movieId]),
                movie_id=record.movieId,
                metadata={**metadata_by_movie_id[movie_id_title[record.movieId]], **metadata_by_user_id[record.userId]},
            ),
            GroundTruth(rating=record.rating),
        )
        for record in df_ratings.itertuples(index=False)
    ]

    print(f"preparing {len(test_samples_and_ground_truths)} test samples")

    datapoints = []
    for record in df_ratings.itertuples(index=False):
        datapoints.append(
            dict(
                user_id=str(record.userId),
                title=movie_id_title[record.movieId],
                movie_id=record.movieId,
                rating=record.rating,
                **metadata_by_movie_id[movie_id_title[record.movieId]],
                **metadata_by_user_id[record.userId],
            ),
        )

    df_dataset = pd.DataFrame.from_records(datapoints)
    dataframe_to_csv(df_dataset, "s3://kolena-public-datasets/movielens/movielens.csv", index=False)

    complete_test_case = TestCase(
        name=f"complete :: {DATASET}-10k",
        description=f"All images in {DATASET} dataset",
        test_samples=test_samples_and_ground_truths,
        reset=True,
    )

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
        test_cases = TestCase.init_many(
            [(f"{cat} :: {type} :: {DATASET}-10k", test_samples) for type, test_samples in ts.items()],
            reset=True,
        )

    TestSuite(
        name=f"{DATASET}-10k :: {type}",
        test_cases=[complete_test_case, *test_cases],
        reset=True,
    )


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
