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


def seed_test_run(model_name: str, detector: str, test_suite_names: List[str]) -> None:
    df = pd.read_csv(f"s3://{BUCKET}/{DATASET}/predictions/predictions_{model_name}.csv")

    def infer(test_sample: TestSample) -> Inference:
        similarities = []
        for img in test_sample.pairs:
            # assume no image pair with itself
            query = df["locator_a"].isin([test_sample.locator, img.locator]) & df["locator_b"].isin(
                [test_sample.locator, img.locator]
            )
            sample_results = df[query]
            similarity = sample_results["similarity"].values[0] if not sample_results["failure"].values[0] else None
            similarities.append(similarity)

        if not df[df["locator_a"] == test_sample.locator].empty:
            r = next(df[df["locator_a"] == test_sample.locator].itertuples(index=False))
            min_x, min_y, max_x, max_y = r.a_min_x, r.a_min_y, r.a_max_x, r.a_max_y
            right_eye_x, right_eye_y = r.a_right_eye_x, r.a_right_eye_y
            left_eye_x, left_eye_y = r.a_left_eye_x, r.a_left_eye_y
            nose_x, nose_y = r.a_nose_x, r.a_nose_y
            mouth_right_x, mouth_right_y = r.a_mouth_right_x, r.a_mouth_right_y
            mouth_left_x, mouth_left_y = r.a_mouth_left_x, r.a_mouth_left_y
        elif not df[df["locator_b"] == test_sample.locator].empty:
            r = next(df[df["locator_b"] == test_sample.locator].itertuples(index=False))
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

    metadata = dict(detector=detector)
    model = Model(f"{model_name} [FR]", infer=infer, metadata=metadata)
    print(f"Model: {model}")

    configurations = [
        ThresholdConfiguration(false_match_rate=1e-1, iou_threshold=0.5, nmse_threshold=0.5),
    ]

    for test_suite_name in test_suite_names:
        test_suite = TestSuite.load(test_suite_name)
        print(f"Test Suite: {test_suite}")
        test(model, test_suite, evaluate_face_recognition_11, configurations, reset=True)


def main(args: Namespace) -> int:
    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)
    for model_name, detector in zip(args.models, args.detectors):
        seed_test_run(model_name, detector, args.test_suites)
    return 0


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--models",
        default=["VGG-Face", "Facenet512"],
        help="Name(s) of model(s) in directory to test",
    )
    ap.add_argument(
        "--detectors",
        default=["MTCNN", "yolov8n-face"],
        help="Name(s) of detectors(s) used with corresponding model(s).",
    )
    ap.add_argument(
        "--test_suites",
        default=[f"{DATASET} :: gender [FR]", f"{DATASET} :: race [FR]"],
        help="Name(s) of test suite(s) to test.",
    )
    sys.exit(main(ap.parse_args()))
