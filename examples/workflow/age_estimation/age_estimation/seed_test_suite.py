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
import sys
from argparse import ArgumentParser
from argparse import Namespace

import pandas as pd
from age_estimation.workflow import GroundTruth
from age_estimation.workflow import TestCase
from age_estimation.workflow import TestSample
from age_estimation.workflow import TestSuite

import kolena

BUCKET = "kolena-public-datasets"
DATASET = "labeled-faces-in-the-wild"


def main(args: Namespace) -> int:
    kolena.initialize(verbose=True)

    df_metadata = pd.read_csv(args.dataset_csv, storage_options={"anon": True})

    non_metadata_fields = {"locator", "age"}
    test_samples_and_ground_truths = [
        (
            TestSample(
                locator=record.locator,
                metadata={f: getattr(record, f) for f in set(record._fields) - non_metadata_fields},
            ),
            GroundTruth(age=record.age),
        )
        for record in df_metadata.itertuples(index=False)
    ]

    # Basic Test Cases
    complete_test_case = TestCase(
        f"complete {DATASET} [age estimation]",
        description=f"All images in {DATASET} dataset with age ground truth",
        test_samples=test_samples_and_ground_truths,
        reset=True,
    )

    # Metadata Test Cases
    age_bins = [(18, 25), (25, 35), (35, 55), (55, 75)]
    test_cases_by_age = [
        TestCase(
            f"age :: ({age_min}, {age_max}] :: {DATASET} [age estimation]",
            description=f"Images in {DATASET} with age between {age_min} (exclusive) and {age_max} (inclusive)",
            test_samples=[(ts, gt) for ts, gt in test_samples_and_ground_truths if age_min < gt.age <= age_max],
            reset=True,
        )
        for age_min, age_max in age_bins
    ]

    test_suite = TestSuite(
        f"age :: {args.suite_name} [age estimation]",
        test_cases=[complete_test_case, *test_cases_by_age],
        reset=True,
    )
    print(f"created test suite: {test_suite}")

    test_cases_by_gender = [
        TestCase(
            f"gender :: {gender} :: {DATASET} [age estimation]",
            description=f"All images in {DATASET} dataset with gender {gender}",
            test_samples=[(ts, gt) for ts, gt in test_samples_and_ground_truths if ts.metadata["gender"] == gender],
            reset=True,
        )
        for gender in ["man", "woman"]
    ]
    test_suite = TestSuite(
        f"gender :: {args.suite_name} [age estimation]",
        test_cases=[complete_test_case, *test_cases_by_gender],
        reset=True,
    )
    print(f"created test suite: {test_suite}")

    races = ["asian", "black", "indian", "latino hispanic", "middle eastern", "white"]
    test_cases_by_race = [
        TestCase(
            f"race :: {race} :: {DATASET} [age estimation]",
            description=f"All images in {DATASET} dataset with race {race}",
            test_samples=[(ts, gt) for ts, gt in test_samples_and_ground_truths if ts.metadata["race"] == race],
            reset=True,
        )
        for race in races
    ]
    test_suite = TestSuite(
        f"race :: {args.suite_name} [age estimation]",
        test_cases=[complete_test_case, *test_cases_by_race],
        reset=True,
    )
    print(f"created test suite: {test_suite}")

    return 0


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset_csv",
        nargs="?",
        default=f"s3://{BUCKET}/{DATASET}/meta/metadata.csv",
        help="CSV file specifying dataset. See default CSV for details",
    )
    ap.add_argument(
        "--test_suite",
        type=str,
        default=DATASET,
        help="Optionally specify a name for the created test suites.",
    )
    sys.exit(main(ap.parse_args()))
