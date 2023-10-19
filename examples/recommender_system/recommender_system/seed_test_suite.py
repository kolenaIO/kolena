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
from typing import List
from typing import Tuple

import pandas as pd
from recommender_system.workflow import GroundTruth
from recommender_system.workflow import TestCase
from recommender_system.workflow import TestSample
from recommender_system.workflow import TestSuite

import kolena

BUCKET = "kolena-public-datasets"
DATASET = "movielens"


def create_test_case_for_genre(
    test_samples_and_ground_truths: List[Tuple[TestCase, GroundTruth]],
    genre: str,
) -> TestCase:
    name = f"genre :: {genre} :: {DATASET}"
    description = f"demographic subset of {DATASET} with source data labeled as genre={genre}"

    # filter down to only test samples matching this genre
    test_samples = [(ts, gt) for ts, gt in test_samples_and_ground_truths if genre in ts.metadata["genres"]]

    test_case = TestCase(
        name=name,
        description=description,
        test_samples=test_samples,
        reset=True,
    )

    return test_case


def main(args: Namespace) -> int:
    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)

    t0 = time.time()

    df_ratings = pd.read_csv(args.ratings_csv)
    df_movies = pd.read_csv(args.movies_csv)

    def process_metadata(record, f):
        value = getattr(record, f)
        return value if f != "genres" else value.split("|")

    metadata_by_movie_id = {}
    for record in df_movies.itertuples(index=False):
        fields = set(record._fields)
        fields.remove("movieId")

        metadata_by_movie_id[record.movieId] = {f: process_metadata(record, f) for f in fields}

    test_samples_and_ground_truths = [
        (
            TestSample(user_id=record.userId, movie_id=record.movieId, metadata=metadata_by_movie_id[record.movieId]),
            GroundTruth(rating=record.rating),
        )
        for record in df_ratings.itertuples(index=False)
    ]

    print(f"prepping {len(test_samples_and_ground_truths)} test samples")

    t1 = time.time()
    complete_test_case = TestCase(
        name=f"ml-100k complete :: {DATASET}",
        description=f"All images in {DATASET} dataset",
        test_samples=test_samples_and_ground_truths,
        reset=True,
    )
    print(f"created baseline test case '{complete_test_case.name}' in {time.time() - t1:0.3f} seconds")

    # Metadata Test Cases
    genre_subsets = [
        "Adventure",
        "Animation",
        "Children",
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
    ]

    genre_ts_gt_splits = {item: [] for item in genre_subsets}

    for ts, gt in test_samples_and_ground_truths:
        for genre in ts.metadata["genres"]:
            genre_ts_gt_splits[genre].append((ts, gt))

    t2 = time.time()
    test_cases = TestCase.init_many(
        [(f"genre :: {genre} :: {DATASET}", test_samples) for genre, test_samples in genre_ts_gt_splits.items()]
    )
    print(f"created test case genre stratifications in {time.time() - t2:0.3f} seconds")

    test_suite = TestSuite(
        name=f"fr 1:1 :: {DATASET}",
        test_cases=[complete_test_case, *test_cases],
        reset=True,
    )
    print(f"created test suite '{test_suite}' in {time.time() - t0:0.3f} seconds")


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--ratings_csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/meta/ratings.100k.csv",
    )
    ap.add_argument(
        "--movies_csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/meta/movies.csv",
    )
    sys.exit(main(ap.parse_args()))
