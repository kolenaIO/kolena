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
from argparse import ArgumentParser
from argparse import Namespace
from ast import literal_eval

import pandas as pd
from semantic_segmentation.constants import DATASET
from semantic_segmentation.constants import PERSON_LABEL
from semantic_segmentation.workflow import GroundTruth
from semantic_segmentation.workflow import TestCase
from semantic_segmentation.workflow import TestSample
from semantic_segmentation.workflow import TestSuite
from tqdm import tqdm

import kolena
from kolena.workflow.annotation import SegmentationMask


def seed_complete_test_case(args: Namespace) -> TestCase:
    df = pd.read_csv(args.dataset_csv)
    test_samples = []
    for record in tqdm(df.itertuples(index=False), total=len(df)):
        test_sample = TestSample(  # type: ignore
            locator=record.locator,
            metadata=dict(
                basename=record.basename,
                annotation_file=record.annotation_file,
                captions=literal_eval(record.captions),
                has_person=record.has_person,
            ),
        )
        ground_truth = GroundTruth(mask=SegmentationMask(locator=record.mask, labels={PERSON_LABEL: "person"}))
        test_samples.append((test_sample, ground_truth))

    test_case = TestCase(f"complete :: {DATASET}", test_samples=test_samples, reset=True)
    print(f"Created test case: {test_case}")

    return test_case


def main(args: Namespace) -> None:
    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)
    complete_test_case = seed_complete_test_case(args)
    test_suite = TestSuite(
        f"{DATASET}",
        test_cases=[complete_test_case],
        reset=True,
    )
    print(f"Created test suite: {test_suite}")


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset_csv",
        type=str,
        default=f"s3://kolena-public-datasets/{DATASET}/annotations/annotations.csv",
        help="CSV file specifying dataset. See default CSV for details",
    )

    main(ap.parse_args())
