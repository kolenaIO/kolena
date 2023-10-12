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
import json
import os
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
from typing import List
from typing import Tuple

import dacite
from object_detection_3d.workflow import GroundTruth
from object_detection_3d.workflow import TestCase
from object_detection_3d.workflow import TestSample
from object_detection_3d.workflow import TestSuite

import kolena


DEFAULT_TEST_SUITE_NAME = "KITTI 3D Object Detection :: training :: metrics"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("sample_file", help="File containing test sample and ground truth data", type=Path)
    parser.add_argument("--test-suite", help="Name of test suite", default=DEFAULT_TEST_SUITE_NAME)
    args = parser.parse_args()
    return args


def seed_test_suite(test_suite_name: str, test_samples: List[Tuple[TestSample, GroundTruth]]):
    kolena.initialize(api_token=os.environ["KOLENA_TOKEN"], verbose=True)

    test_cases = TestCase.init_many([(test_suite_name, test_samples)], reset=True)
    test_suite = TestSuite(test_suite_name, test_cases=test_cases, reset=True)
    print(f"created test suite {test_suite.name}")


def main(args):
    with open(str(args.sample_file)) as f:
        data = json.load(f)["data"]

    config = dacite.Config(cast=[Enum, tuple], check_types=False)
    test_samples_and_ground_truths = [
        (dacite.from_dict(TestSample, test_sample, config), dacite.from_dict(GroundTruth, ground_truth, config))
        for test_sample, ground_truth in data
    ]

    seed_test_suite(args.test_suite, test_samples_and_ground_truths)


if __name__ == "__main__":
    args = parse_args()
    main(args)
