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
from collections import defaultdict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from face_recognition_11.workflow import GroundTruth
from face_recognition_11.workflow import TestCase
from face_recognition_11.workflow import TestSample
from face_recognition_11.workflow import TestSuite

import kolena
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import Keypoints
from kolena.workflow.asset import ImageAsset

BUCKET = "kolena-public-datasets"
DATASET = "labeled-faces-in-the-wild"


def create_test_case_for_tag(
    test_samples_and_ground_truths: List[Tuple[TestCase, GroundTruth]],
    category: str,
    value: str,
) -> TestCase:
    name = f"{category} :: {value} [FR]"
    description = f"demographic subset of {DATASET} with source data labeled as {category}={value}"

    # filter down to only test samples matching this demographic
    test_samples = [(ts, gt) for ts, gt in test_samples_and_ground_truths if ts.metadata[category] == value]

    test_case = TestCase(
        name=name,
        description=description,
        test_samples=test_samples,
        reset=True,
    )

    return test_case


def main(args: Namespace) -> int:
    kolena.initialize(verbose=True)

    df = pd.read_csv(args.dataset_csv)
    df_metadata = pd.read_csv(args.metadata_csv)
    df_bbox_keypoints = pd.read_csv(args.bbox_keypoints_csv)

    metadata_by_locator = {}
    locator_normalization_factor = {}
    non_metadata_fields = {"locator", "normalization_factor"}
    for record in df_metadata.itertuples(index=False):
        fields = set(record._fields)
        metadata_by_locator[record.locator] = {f: getattr(record, f) for f in fields - non_metadata_fields}
        locator_normalization_factor[record.locator] = record.normalization_factor

    bboxes = {}
    keypoints = {}
    for record in df_bbox_keypoints.itertuples(index=False):
        fields = set(record._fields)
        bboxes[record.locator] = BoundingBox(
            top_left=(record.min_x, record.min_y),
            bottom_right=(record.max_x, record.max_y),
        )
        keypoints[record.locator] = Keypoints(
            points=[
                (record.right_eye_x, record.right_eye_y),
                (record.left_eye_x, record.left_eye_y),
                (record.nose_x, record.nose_y),
                (record.mouth_right_x, record.mouth_right_y),
                (record.mouth_left_x, record.mouth_left_y),
            ],
        )

    images = defaultdict(list)
    for record in df.itertuples(index=False):
        images[record.locator_a].append(record.locator_b)
        images[record.locator_b].append(record.locator_a)

    test_samples_and_ground_truths = list()
    for locator, pairs in images.items():
        ts = TestSample(locator=locator, pairs=[ImageAsset(p) for p in pairs], metadata=metadata_by_locator[locator])

        matches = []
        for img in pairs:
            # assume no image pair with itself
            query = df["locator_a"].isin([locator, img]) & df["locator_b"].isin([locator, img])
            match = df[query]["is_same"].values[0]
            matches.append(match)

        gt = GroundTruth(
            matches=matches,
            bbox=bboxes[locator],
            keypoints=keypoints[locator],
            normalization_factor=locator_normalization_factor[locator],
            count_genuine_pair=np.sum(matches),
            count_imposter_pair=np.sum(np.invert(matches)),
        )
        test_samples_and_ground_truths.append((ts, gt))

    complete_test_case = TestCase(
        name=f"{DATASET} :: complete [FR]",
        description=f"All images in {DATASET} dataset",
        test_samples=test_samples_and_ground_truths,
        reset=True,
    )

    # Metadata Test Cases
    demographic_subsets = dict(
        race=["asian", "black", "indian", "middle eastern", "latino hispanic", "white"],  # ignore "unknown"
        gender=["man", "woman"],  # ignore "unknown"
    )

    test_suites = defaultdict(list)
    for category, tags in demographic_subsets.items():
        test_cases = []
        for tag in tags:
            test_case = create_test_case_for_tag(test_samples_and_ground_truths, category, tag)
            test_cases.append(test_case)
            print(f"created test case '{test_case.name}'")
        test_suites[category] = test_cases

    for test_suite_name, test_cases in test_suites.items():
        test_suite = TestSuite(
            name=f"{DATASET} :: {test_suite_name} [FR]",
            test_cases=[complete_test_case, *test_cases],
            reset=True,
        )
        print(f"created test suite '{test_suite}'")


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset_csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/meta/pairs.sample.csv",
        help="CSV file containing image pairs to be tested. See default CSV for details.",
    )
    ap.add_argument(
        "--bbox_keypoints_csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/meta/bbox_keypoints.csv",
        help="CSV file containing bbox and keypoints for each image. See default CSV for details.",
    )
    ap.add_argument(
        "--metadata_csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/meta/metadata.csv",
        help="CSV file containing the metadata of each image. See default CSV for details.",
    )
    sys.exit(main(ap.parse_args()))
