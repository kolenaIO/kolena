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
import itertools
import os
import sys
from argparse import ArgumentParser
from argparse import Namespace
from typing import List

import pandas as pd

from face_recognition_11.workflow import GroundTruth
from face_recognition_11.workflow import TestCase
from face_recognition_11.workflow import TestSample
from face_recognition_11.workflow import TestSuite
from face_recognition_11.workflow import ImageWithMetadata

import kolena

BUCKET = "kolena-public-datasets"
DATASET = "labeled-faces-in-the-wild"


def main(args: Namespace) -> int:
    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)

    df = pd.read_csv(args.dataset_csv)
    df_metadata = pd.read_csv(args.metadata_csv)

    metadata_by_locator = {}
    for record in df_metadata.itertuples(index=False):
        fields = set(record._fields)
        fields.remove("locator")
        metadata_by_locator[record.locator] = {f: getattr(record, f) for f in fields}

    test_samples = [
        TestSample(
            a=ImageWithMetadata(locator=row["locator_a"], metadata=metadata_by_locator[row["locator_a"]]),
            b=ImageWithMetadata(locator=row["locator_b"], metadata=metadata_by_locator[row["locator_b"]]),
        )
        for idx, row in df[["locator_a", "locator_b"]].iterrows()
    ]
    ground_truths = [GroundTruth(is_same=(ts.a.metadata["person"] == ts.b.metadata["person"])) for ts in test_samples]

    test_samples_and_ground_truths = list(zip(test_samples, ground_truths))

    complete_test_case = TestCase(
        f"fr 1:1 complete :: {DATASET}",
        description=f"All images in {DATASET} dataset",
        test_samples=test_samples_and_ground_truths,
        reset=True,
    )

    # test_case = TestCase(name=f"fr 1:1 :: {DATASET} test case", test_samples=test_samples_and_ground_truths, reset=True)

    section_size = 2500
    splits = [
        test_samples_and_ground_truths[:section_size],
        test_samples_and_ground_truths[section_size : 2 * section_size],
        test_samples_and_ground_truths[2 * section_size : 3 * section_size],
        test_samples_and_ground_truths[3 * section_size :],
    ]

    test_cases: List[TestCase] = []
    for idx, split in enumerate(splits):
        test_cases.append(
            TestCase(
                f"fr 1:1 :: split {idx} :: {DATASET}",
                description=f"Images in {DATASET} in split {idx}",
                test_samples=[(ts, gt) for ts, gt in split],
                reset=True,
            ),
        )

    test_suite = TestSuite(
        name=f"fr 1:1 :: {DATASET}",
        test_cases=[complete_test_case, *test_cases],
        reset=True,
    )
    print(f"created test suite: {test_suite}")


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset_csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/meta/pairs.sample.csv",
        help="CSV file containing image pairs to be tested. See default CSV for details.",
    )
    ap.add_argument(
        "--metadata_csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/meta/metadata.csv",
        help="CSV file containing the metadata of each image. See default CSV for details.",
    )
    sys.exit(main(ap.parse_args()))
