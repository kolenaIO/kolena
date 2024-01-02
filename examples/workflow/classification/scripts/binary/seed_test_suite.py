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
from typing import List

import pandas as pd
from classification.workflow import GroundTruth
from classification.workflow import TestCase
from classification.workflow import TestSample
from classification.workflow import TestSuite

import kolena
from kolena.workflow.annotation import ClassificationLabel

BUCKET = "kolena-public-datasets"
DATASET = "dogs-vs-cats"
POSITIVE_LABEL = "dog"


def main(args: Namespace) -> int:
    kolena.initialize(verbose=True)

    df_metadata = pd.read_csv(args.dataset_csv, storage_options={"anon": True})

    non_metadata_fields = {"locator", "label"}
    test_samples_and_ground_truths = [
        (
            TestSample(
                locator=record.locator,
                metadata={f: getattr(record, f) for f in set(record._fields) - non_metadata_fields},
            ),
            GroundTruth(classification=ClassificationLabel(record.label)),
        )
        for record in df_metadata.itertuples(index=False)
    ]

    # Basic Test Cases
    complete_test_case = TestCase(
        f"complete {DATASET}",
        description=f"All images in {DATASET} dataset",
        test_samples=test_samples_and_ground_truths,
        reset=True,
    )

    # Metadata Test Cases
    stratification_logic_map = {
        "small": lambda area: 0 <= area < 80000,
        "medium": lambda area: 80000 <= area < 160000,
        "large": lambda area: area >= 160000,
    }

    test_cases: List[TestCase] = []
    for name, fn in stratification_logic_map.items():
        test_cases.append(
            TestCase(
                f"image size :: {name} :: {DATASET}",
                description=f"Images in {DATASET} with {name} image size",
                test_samples=[
                    (ts, gt)
                    for ts, gt in test_samples_and_ground_truths
                    if fn(ts.metadata["width"] * ts.metadata["height"])
                ],
                reset=True,
            ),
        )

    test_suite = TestSuite(
        f"image size :: {DATASET}",
        test_cases=[complete_test_case, *test_cases],
        reset=True,
    )
    print(f"created test suite: {test_suite}")

    return 0


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset_csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/meta/metadata.csv",
        help="CSV file with a list of image `locator` and its `label`. See default CSV for details",
    )
    sys.exit(main(ap.parse_args()))
