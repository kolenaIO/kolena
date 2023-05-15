import os

import pandas as pd
from kolena_contrib.age_estimation.workflow import GroundTruth
from kolena_contrib.age_estimation.workflow import TestCase
from kolena_contrib.age_estimation.workflow import TestSample
from kolena_contrib.age_estimation.workflow import TestSuite

import kolena

BUCKET = "kolena-public-datasets"
DATASET = "labeled-faces-in-the-wild"
LOCAL_DIR = "/data/open-source/lfw"

env_token = "KOLENA_TOKEN"
print(f"initializing with environment variables ${env_token}")
kolena.initialize(os.environ[env_token], verbose=True)

df_metadata = pd.read_csv(f"s3://{BUCKET}/{DATASET}/meta/metadata.csv")

test_samples_and_ground_truths = [
    (
        TestSample(  # type: ignore
            locator=record.locator,
            name=record.person,
            race=record.race,
            gender=record.gender,
            age=record.age,
            image_width=record.width,
            image_height=record.height,
        ),
        GroundTruth(  # type: ignore
            age=record.age,
        ),
    )
    for record in df_metadata.itertuples()
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
        test_samples=[(ts, gt) for ts, gt in test_samples_and_ground_truths if ts.gender == gender],
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
        test_samples=[(ts, gt) for ts, gt in test_samples_and_ground_truths if ts.race == race],
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
