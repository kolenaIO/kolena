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

import pandas as pd
from face_recognition_11.evaluator import evaluate_face_recognition_11
from face_recognition_11.workflow import Inference
from face_recognition_11.workflow import Model
from face_recognition_11.workflow import TestSample
from face_recognition_11.workflow import TestSuite
from face_recognition_11.workflow import ThresholdConfiguration

import kolena
from kolena.workflow import test
from kolena.workflow.annotation import BoundingBox, Keypoints

BUCKET = "kolena-public-datasets"
DATASET = "labeled-faces-in-the-wild"


def seed_test_run(model_name: str, test_suite_names: List[str]) -> None:
    df_results = pd.read_csv(f"s3://{BUCKET}/{DATASET}/predictions/predictions_{model_name}.csv")

    def infer(test_sample: TestSample) -> Inference:
        # TODO: Dummy inferences -- fix
        return Inference(
            left_bbox=BoundingBox((0, 0), (0, 0)),
            left_keypoints=Keypoints([(0, 0)]),
            right_bbox=BoundingBox((0, 0), (0, 0)),
            right_keypoints=Keypoints([(0, 0)]),
            similarity=0.9,
        )

    model = Model(f"{model_name} [{DATASET}]", infer=infer)
    print(f"Model: {model}")

    for test_suite_name in test_suite_names:
        test_suite = TestSuite.load(test_suite_name)
        print(f"Test Suite: {test_suite}")

        test(
            model,
            test_suite,
            evaluate_face_recognition_11,
            configurations=[
                ThresholdConfiguration(threshold=0.3),
            ],
            reset=True,
        )


def main(args: Namespace) -> int:
    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)
    for model_name in args.models:
        seed_test_run(model_name, args.test_suites)
    return 0


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--models",
        # default=["Paravision"],
        default=["deepface"],
        help="Name(s) of model(s) in directory to test",
    )
    ap.add_argument(
        "--test_suites",
        default=[f"fr 1:1 :: {DATASET}"],
        help="Name(s) of test suite(s) to test.",
    )
    sys.exit(main(ap.parse_args()))
