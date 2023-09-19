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

    non_metadata_fields = ["locator"]
    df_metadata = df.drop(non_metadata_fields, axis=1)
    metadata_by_person = {
        record.person: {f: getattr(record, f) for f in set(record._fields)}
        for record in df_metadata.itertuples(index=False)
    }

    # form pairs from image locators & metadata
    image_pairs = list(
        itertools.combinations(
            [(locator, metadata_by_person[person]) for person, locator in zip(df["person"], df["locator"])], 2
        )
    )

    # an image pair will contain a left and right tuple with the respective locator & metadata
    test_samples = [
        TestSample(a=ImageWithMetadata(locator=a[0], metadata=a[1]), b=ImageWithMetadata(locator=b[0], metadata=b[1]))
        for a, b in image_pairs
    ]
    ground_truths = [GroundTruth(is_same=(ts.a.metadata["person"] == ts.a.metadata["person"])) for ts in test_samples]

    test_samples_and_ground_truths = list(zip(test_samples, ground_truths))
    test_case = TestCase(name=f"fr 1:1 :: {DATASET} test case", test_samples=test_samples_and_ground_truths, reset=True)

    test_suite = TestSuite(
        name=f"fr 1:1 :: {DATASET}",
        test_cases=[test_case],
        reset=True,
    )
    print(f"created test suite: {test_suite}")


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset_csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/meta/metadata.tiny5.csv",
        help="CSV file with a list of image `locator` and its `label`. See default CSV for details",
    )
    sys.exit(main(ap.parse_args()))
