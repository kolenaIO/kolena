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
from argparse import ArgumentParser
from argparse import Namespace

import pandas as pd
from keypoint_detection.workflow import GroundTruth
from keypoint_detection.workflow import TestCase
from keypoint_detection.workflow import TestSample
from keypoint_detection.workflow import TestSuite

import kolena
from kolena.workflow.annotation import Keypoints

DATASET = "300-W"
BUCKET = "s3://kolena-public-datasets"


def run(args: Namespace) -> None:
    df = pd.read_csv(f"{BUCKET}/{DATASET}/meta/metadata.csv", storage_options={"anon": True})

    test_samples = [TestSample(locator) for locator in df["locator"]]
    ground_truths = [
        GroundTruth(
            face=Keypoints(points=json.loads(record.points)),
            normalization_factor=record.normalization_factor,
        )
        for record in df.itertuples()
    ]
    ts_with_gt = list(zip(test_samples, ground_truths))
    complete_test_case = TestCase(args.test_suite + " test case", test_samples=ts_with_gt, reset=True)

    TestSuite(args.test_suite, test_cases=[complete_test_case], reset=True)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("test_suite", type=str, help="Name of the test suite to make.")

    kolena.initialize(verbose=True)

    run(ap.parse_args())


if __name__ == "__main__":
    main()
