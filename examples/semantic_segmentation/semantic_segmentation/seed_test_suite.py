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
from typing import List
from typing import Tuple

import pandas as pd
from semantic_segmentation.constants import DATASET
from semantic_segmentation.constants import PERSON_COUNT_MAPPING_IMAGES
from semantic_segmentation.workflow import GroundTruth
from semantic_segmentation.workflow import TestCase
from semantic_segmentation.workflow import TestSample
from semantic_segmentation.workflow import TestSuite
from tqdm import tqdm

import kolena
from kolena.workflow.annotation import SegmentationMask


def within_range(area: int, range: Tuple[int, int]) -> bool:
    return range[0] <= area < range[1]


def seed_stratified_test_cases(complete_test_case: TestCase, test_suite_name) -> List[TestCase]:
    test_samples = complete_test_case.load_test_samples()
    test_cases = []
    for name, count_range in PERSON_COUNT_MAPPING_IMAGES.items():
        samples = []
        for ts, gt in test_samples[:50]:
            person_count = ts.metadata["person_count"]
            if within_range(person_count, count_range):
                samples.append((ts, gt))

        if len(samples) > 0:
            test_cases.append(
                TestCase(f"{name} :: {test_suite_name}", test_samples=samples, reset=True),
            )
    return test_cases


def seed_complete_test_case(args: Namespace) -> TestCase:
    df = pd.read_csv(args.dataset_csv)
    test_samples = []
    for record in tqdm(df.itertuples(index=False), total=50):
        test_sample = TestSample(  # type: ignore
            locator=record.locator,
            metadata=dict(
                basename=record.basename,
                annotation_file=record.annotation_file,
                captions=literal_eval(record.captions),
                has_person=record.has_person,
                width=record.image_width,
                height=record.image_height,
                person_count=record.person_count,
            ),
        )
        ground_truth = GroundTruth()
        test_samples.append((test_sample, ground_truth))

        if len(test_samples) == 50:
            break

    test_case = TestCase(f"complete :: {DATASET} [person] 5", test_samples=test_samples, reset=True)
    return test_case


def main(args: Namespace) -> None:
    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)
    test_suite_name = f"# of people :: {DATASET} [person] 5"
    complete_test_case = seed_complete_test_case(args)
    stratified_test_cases = seed_stratified_test_cases(complete_test_case, test_suite_name)
    TestSuite(
        test_suite_name,
        test_cases=[complete_test_case] + stratified_test_cases,
        reset=True,
    )


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset_csv",
        type=str,
        default=f"s3://kolena-public-datasets/{DATASET}/annotations/annotations_person.csv",
        help="CSV file specifying dataset. See default CSV for details",
    )

    main(ap.parse_args())
