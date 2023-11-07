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
from face_recognition_11.workflow import ThresholdConfiguration
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
    df_results = pd.read_csv("preds.sample.csv")

    def infer(test_sample: TestSample) -> Inference:
        similarities = []
        for img in test_sample.pairs:
            sample_results = df_results[
                ((df_results["locator_a"] == test_sample.locator) & (df_results["locator_b"] == img.locator))
                | ((df_results["locator_a"] == img.locator) & (df_results["locator_b"] == test_sample.locator))
            ]
            similarity = sample_results["similarity"].values[0] if not sample_results["failure"].values[0] else None
            similarities.append(similarity)

        if not df_results[df_results["locator_a"] == test_sample.locator].empty:
            r = next(df_results[df_results["locator_a"] == test_sample.locator].itertuples(index=False))
            min_x, min_y, max_x, max_y = r.a_min_x, r.a_min_y, r.a_max_x, r.a_max_y
            right_eye_x, right_eye_y = r.a_right_eye_x, r.a_right_eye_y
            left_eye_x, left_eye_y = r.a_left_eye_x, r.a_left_eye_y
            nose_x, nose_y = r.a_nose_x, r.a_nose_y
            mouth_right_x, mouth_right_y = r.a_mouth_right_x, r.a_mouth_right_y
            mouth_left_x, mouth_left_y = r.a_mouth_left_x, r.a_mouth_left_y
        elif not df_results[df_results["locator_b"] == test_sample.locator].empty:
            r = next(df_results[df_results["locator_b"] == test_sample.locator].itertuples(index=False))
            min_x, min_y, max_x, max_y = r.b_min_x, r.b_min_y, r.b_max_x, r.b_max_y
            right_eye_x, right_eye_y = r.b_right_eye_x, r.b_right_eye_y
            left_eye_x, left_eye_y = r.b_left_eye_x, r.b_left_eye_y
            nose_x, nose_y = r.b_nose_x, r.b_nose_y
            mouth_right_x, mouth_right_y = r.b_mouth_right_x, r.b_mouth_right_y
            mouth_left_x, mouth_left_y = r.b_mouth_left_x, r.b_mouth_left_y

        bbox = BoundingBox((min_x, min_y), (max_x, max_y)) if min_x else None
        keypoints = (
            Keypoints(
                [
                    (right_eye_x, right_eye_y),
                    (left_eye_x, left_eye_y),
                    (nose_x, nose_y),
                    (mouth_right_x, mouth_right_y),
                    (mouth_left_x, mouth_left_y),
                ],
            )
            if right_eye_x
            else None
        )

        return Inference(similarities=similarities, bbox=bbox, keypoints=keypoints)

    model = Model(f"{model_name} [{DATASET}]", infer=infer)
    print(f"Model: {model}")

    configurations = [
        ThresholdConfiguration(false_match_rate=1e-1, iou_threshold=0.5),
        # ThresholdConfiguration(false_match_rate=1e-2),
        # ThresholdConfiguration(false_match_rate=1e-3),
        # ThresholdConfiguration(false_match_rate=1e-4),
    ]

    for test_suite_name in test_suite_names:
        test_suite = TestSuite.load(test_suite_name)
        print(f"Test Suite: {test_suite}")
        test(model, test_suite, evaluate_face_recognition_11, configurations, reset=True)


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
        default=[f"fr 1:1 holistic :: {DATASET}"],
        help="Name(s) of test suite(s) to test.",
    )
    sys.exit(main(ap.parse_args()))
