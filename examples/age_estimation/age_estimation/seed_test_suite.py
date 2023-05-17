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

import pandas as pd
from age_estimation.workflow import GroundTruth
from age_estimation.workflow import TestCase
from age_estimation.workflow import TestSample
from age_estimation.workflow import TestSuite

import kolena


BUCKET = "kolena-public-datasets"
DATASET = "labeled-faces-in-the-wild"

env_token = "KOLENA_TOKEN"
print(f"initializing with environment variables ${env_token}")
kolena.initialize(os.environ[env_token], verbose=True)

df_metadata = pd.read_csv(f"s3://{BUCKET}/{DATASET}/meta/metadata.csv")

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
test_cases_by_age = []
test_cases_by_age.append(
    TestCase(
        f"age :: (18, 25] :: {DATASET} [age estimation]",
        description=f"All images in {DATASET} dataset with age in between 18 - 25",
        test_samples=[(ts, gt) for ts, gt in test_samples_and_ground_truths if 18 < gt.age <= 25],
        reset=True,
    ),
)
test_cases_by_age.append(
    TestCase(
        f"age :: (25, 35] :: {DATASET} [age estimation]",
        description=f"All images in {DATASET} dataset with age in between 25 - 35",
        test_samples=[(ts, gt) for ts, gt in test_samples_and_ground_truths if 25 < gt.age <= 35],
        reset=True,
    ),
)
test_cases_by_age.append(
    TestCase(
        f"age :: (35, 55] :: {DATASET} [age estimation]",
        description=f"All images in {DATASET} dataset with age in between 35 - 55",
        test_samples=[(ts, gt) for ts, gt in test_samples_and_ground_truths if 35 < gt.age <= 55],
        reset=True,
    ),
)
test_cases_by_age.append(
    TestCase(
        f"age :: (55, 75] {DATASET} [age estimation]",
        description=f"All images in {DATASET} dataset with age in between 55 - 75",
        test_samples=[(ts, gt) for ts, gt in test_samples_and_ground_truths if 55 < gt.age <= 75],
        reset=True,
    ),
)

test_suite = TestSuite(
    f"complete {DATASET} [age estimation]",
    test_cases=[complete_test_case, *test_cases_by_age],
    reset=True,
)
print(f"created test suite {test_suite}")

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
    f"gender :: {DATASET} [age estimation]",
    test_cases=[complete_test_case, *test_cases_by_gender],
    reset=True,
)
print(f"created test suite {test_suite}")

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
    f"race :: {DATASET} [age estimation]",
    test_cases=[complete_test_case, *test_cases_by_race],
    reset=True,
)
print(f"created test suite {test_suite}")
