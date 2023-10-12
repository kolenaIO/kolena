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
from face_recognition_11.workflow import FMRConfiguration
from face_recognition_11.workflow import Inference
from face_recognition_11.workflow import Model
from face_recognition_11.workflow import TestSample
from face_recognition_11.workflow import TestSuite

import kolena
from kolena.workflow import test
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import Keypoints

BUCKET = "kolena-public-datasets"
DATASET = "labeled-faces-in-the-wild"


def seed_test_run(model_name: str, test_suite_names: List[str]) -> None:
    df_results = pd.read_csv(f"s3://{BUCKET}/{DATASET}/predictions/predictions_{model_name}.sample.csv")

    def infer(test_sample: TestSample) -> Inference:
        sample_results = df_results[
            (df_results["locator_a"] == test_sample.a.locator) & (df_results["locator_b"] == test_sample.b.locator)
        ].iloc[0]

        similarity = None if sample_results["failure"] else sample_results["similarity"]

        return Inference(
            a_bbox=BoundingBox(
                (sample_results["a_min_x"], sample_results["a_min_y"]),
                (sample_results["a_max_x"], sample_results["a_max_y"]),
            ),
            a_keypoints=Keypoints(
                [
                    (sample_results["a_right_eye_x"], sample_results["a_right_eye_y"]),
                    (sample_results["a_left_eye_x"], sample_results["a_left_eye_y"]),
                    (sample_results["a_nose_x"], sample_results["a_nose_y"]),
                    (sample_results["a_mouth_right_x"], sample_results["a_mouth_right_y"]),
                    (sample_results["a_mouth_left_x"], sample_results["a_mouth_left_y"]),
                ],
            ),
            b_bbox=BoundingBox(
                (sample_results["b_min_x"], sample_results["b_min_y"]),
                (sample_results["b_max_x"], sample_results["b_max_y"]),
            ),
            b_keypoints=Keypoints(
                [
                    (sample_results["b_right_eye_x"], sample_results["b_right_eye_y"]),
                    (sample_results["b_left_eye_x"], sample_results["b_left_eye_y"]),
                    (sample_results["b_nose_x"], sample_results["b_nose_y"]),
                    (sample_results["b_mouth_right_x"], sample_results["b_mouth_right_y"]),
                    (sample_results["b_mouth_left_x"], sample_results["b_mouth_left_y"]),
                ],
            ),
            similarity=similarity,
        )

    model = Model(f"{model_name} [{DATASET}]", infer=infer)
    print(f"Model: {model}")

    configurations = [
        FMRConfiguration(false_match_rate=1e-1),
        FMRConfiguration(false_match_rate=1e-2),
        FMRConfiguration(false_match_rate=1e-3),
        FMRConfiguration(false_match_rate=1e-4),
        FMRConfiguration(false_match_rate=1e-5),
        FMRConfiguration(false_match_rate=1e-6),
    ]

    for test_suite_name in test_suite_names:
        test_suite = TestSuite.load(test_suite_name)
        print(f"Test Suite: {test_suite}")
        test(model, test_suite, evaluate_face_recognition_11, configurations)


def main(args: Namespace) -> int:
    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)
    for model_name in args.models:
        seed_test_run(model_name, args.test_suites)
    return 0


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--models",
        default=["deepface"],
        help="Name(s) of model(s) in directory to test",
    )
    ap.add_argument(
        "--test_suites",
        default=[f"fr 1:1 :: {DATASET}"],
        help="Name(s) of test suite(s) to test.",
    )
    sys.exit(main(ap.parse_args()))
