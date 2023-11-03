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
from typing import Tuple

import pandas as pd
from face_recognition_11.workflow import GroundTruth
from face_recognition_11.workflow import TestCase
from face_recognition_11.workflow import TestSample
from face_recognition_11.workflow import TestSuite
from face_recognition_11.workflow import SingleImageGroundTruth

from kolena.workflow import Image
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import Keypoints

import kolena

BUCKET = "kolena-public-datasets"
DATASET = "labeled-faces-in-the-wild"


def create_test_case_for_tag(
    test_samples_and_ground_truths: List[Tuple[TestCase, GroundTruth]],
    category: str,
    value: str,
) -> TestCase:
    name = f"{category} :: {value} :: {DATASET}"
    description = f"demographic subset of {DATASET} with source data labeled as {category}={value}"

    # filter down to only test samples matching this demographic
    test_samples = [
        (ts, gt)
        for ts, gt in test_samples_and_ground_truths
        if ts.metadata["a_" + category] == value or ts.metadata["b_" + category] == value
    ]

    test_case = TestCase(
        name=name,
        description=description,
        test_samples=test_samples,
        reset=True,
    )

    return test_case


def main(args: Namespace) -> int:
    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)

    df = pd.read_csv(args.dataset_csv)
    df_metadata = pd.read_csv(args.metadata_csv)

    metadata_by_locator = {}
    for record in df_metadata.itertuples(index=False):
        fields = set(record._fields)
        fields.remove("locator")
        metadata_by_locator[record.locator] = {f: getattr(record, f) for f in fields}

    test_samples_and_ground_truths = [
        (
            TestSample(
                a=Image(locator=row["locator_a"]),
                b=Image(locator=row["locator_b"]),
                metadata={
                    **{"a_" + k: v for k, v in metadata_by_locator[row["locator_a"]].items()},
                    **{"b_" + k: v for k, v in metadata_by_locator[row["locator_b"]].items()},
                },
            ),
            GroundTruth(
                is_same=row["is_same"],
                a=SingleImageGroundTruth(
                    bbox=BoundingBox(
                        (row["a_min_x"], row["a_min_y"]),
                        (row["a_max_x"], row["a_max_y"]),
                    ),
                    keypoints=Keypoints(
                        [
                            (row["a_right_eye_x"], row["a_right_eye_y"]),
                            (row["a_left_eye_x"], row["a_left_eye_y"]),
                            (row["a_nose_x"], row["a_nose_y"]),
                            (row["a_mouth_right_x"], row["a_mouth_right_y"]),
                            (row["a_mouth_left_x"], row["a_mouth_left_y"]),
                        ],
                    ),
                ),
                b=SingleImageGroundTruth(
                    bbox=BoundingBox(
                        (row["b_min_x"], row["b_min_y"]),
                        (row["b_max_x"], row["b_max_y"]),
                    ),
                    keypoints=Keypoints(
                        [
                            (row["b_right_eye_x"], row["b_right_eye_y"]),
                            (row["b_left_eye_x"], row["b_left_eye_y"]),
                            (row["b_nose_x"], row["b_nose_y"]),
                            (row["b_mouth_right_x"], row["b_mouth_right_y"]),
                            (row["b_mouth_left_x"], row["b_mouth_left_y"]),
                        ],
                    ),
                ),
            ),
        )
        for _, row in df.iterrows()
    ]

    complete_test_case = TestCase(
        name=f"fr 1:1 holistic complete :: {DATASET}",
        description=f"All images in {DATASET} dataset",
        test_samples=test_samples_and_ground_truths,
        reset=True,
    )
    print(f"created baseline test case '{complete_test_case.name}'")

    # Metadata Test Cases
    # demographic_subsets = dict(
    #     race=["asian", "black", "indian", "middle eastern", "latino hispanic", "white"],  # ignore "unknown"
    #     gender=["man", "woman"],  # ignore "unknown"
    # )

    demographic_subsets = dict(
        gender=["man"],
    )

    test_cases: List[TestCase] = []
    for category, tags in demographic_subsets.items():
        for tag in tags:
            test_case = create_test_case_for_tag(test_samples_and_ground_truths, category, tag)
            test_cases.append(test_case)
            print(f"created test case '{test_case.name}'")

    test_suite = TestSuite(
        name=f"fr 1:1 holistic :: {DATASET}",
        test_cases=[complete_test_case, *test_cases],
        reset=True,
    )
    print(f"created test suite '{test_suite}'")


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset_csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/meta/pairs.50.csv",
        help="CSV file containing image pairs to be tested. See default CSV for details.",
    )
    ap.add_argument(
        "--metadata_csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/meta/metadata.csv",
        help="CSV file containing the metadata of each image. See default CSV for details.",
    )
    sys.exit(main(ap.parse_args()))
