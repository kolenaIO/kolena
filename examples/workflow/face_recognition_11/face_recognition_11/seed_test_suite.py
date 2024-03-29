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
from collections import defaultdict
from typing import DefaultDict
from typing import Dict
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
TestCaseSubsetDict = DefaultDict[str, Dict[str, List[Tuple[TestSample, GroundTruth]]]]


def main(args: Namespace) -> int:
    kolena.initialize(verbose=True)

    df = pd.read_csv(args.dataset_csv, storage_options={"anon": True})
    df_metadata = pd.read_csv(args.metadata_csv, storage_options={"anon": True})
    df_bbox_keypoints = pd.read_csv(args.bbox_keypoints_csv, storage_options={"anon": True})

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
        bboxes[record.locator] = BoundingBox(
            top_left=(record.min_x, record.min_y),
            bottom_right=(record.max_x, record.max_y),
        )
        keypoints[record.locator] = Keypoints(
            points=[
                (record.right_eye_x, record.right_eye_y),
                (record.left_eye_x, record.left_eye_y),
                (record.nose_x, record.nose_y),
                (record.right_mouth_x, record.right_mouth_y),
                (record.left_mouth_x, record.left_mouth_y),
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
        test_samples=test_samples_and_ground_truths,  # type: ignore
        reset=True,
    )

    # Metadata Test Cases
    demographic_subsets = dict(
        race=["asian", "black", "indian", "middle eastern", "latino hispanic", "white"],  # ignore "unknown"
        gender=["man", "woman"],  # ignore "unknown"
    )
    test_case_subsets: TestCaseSubsetDict = defaultdict(lambda: defaultdict(list))

    for ts, gt in test_samples_and_ground_truths:
        for category, tags in demographic_subsets.items():
            for tag in tags:
                if ts.metadata[category] == tag:
                    test_case_subsets[category][tag].append((ts, gt))

    for category, tags_dict in test_case_subsets.items():
        test_cases = []
        for tag, test_samples in tags_dict.items():
            name = f"{category} :: {tag} [FR]"
            description = f"demographic subset of {DATASET} with source data labeled as {category}={tag}"
            test_cases.append(
                TestCase(
                    name=name,
                    description=description,
                    test_samples=test_samples,  # type: ignore
                    reset=True,
                ),
            )
        TestSuite(
            name=f"{args.test_suite} :: {category} [FR]",
            test_cases=[complete_test_case, *test_cases],
            reset=True,
        )
    return 0


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset-csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/meta/pairs.30k.csv",
        help="CSV file containing image pairs to be tested. See default CSV for details.",
    )
    ap.add_argument(
        "--bbox-keypoints-csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/meta/bbox_keypoints.30k.csv",
        help="CSV file containing bbox and keypoints for each image. See default CSV for details.",
    )
    ap.add_argument(
        "--metadata-csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/meta/metadata.csv",
        help="CSV file containing the metadata of each image. See default CSV for details.",
    )
    ap.add_argument(
        "--test-suite",
        type=str,
        default=DATASET,
        help="Optionally specify a name for the created test suite.",
    )
    sys.exit(main(ap.parse_args()))
