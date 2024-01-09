# Copyright 2021-2024 Kolena Inc.
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

import numpy as np
import pandas as pd
from face_recognition_11.evaluator import evaluate_face_recognition_11
from face_recognition_11.workflow import Inference
from face_recognition_11.workflow import Model
from face_recognition_11.workflow import TestSample
from face_recognition_11.workflow import TestSuite
from face_recognition_11.workflow import ThresholdConfiguration

import kolena
from kolena.workflow import test
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import Keypoints

BUCKET = "kolena-public-datasets"
DATASET = "labeled-faces-in-the-wild"


def seed_test_run(model_name: str, detector: str, test_suite_names: List[str]) -> None:
    df = pd.read_csv(
        f"s3://{BUCKET}/{DATASET}/predictions/predictions_{model_name}.csv",
        storage_options={"anon": True},
    )

    def infer(test_sample: TestSample) -> Inference:
        similarities = []
        for img in test_sample.pairs:
            # assume no image pair with itself
            query = df["locator_a"].isin([test_sample.locator, img.locator]) & df["locator_b"].isin(
                [test_sample.locator, img.locator],
            )
            sample_results = df[query]
            similarity = sample_results["similarity"].values[0] if not sample_results["failure"].values[0] else None
            similarities.append(similarity)

        if not df[df["locator_a"] == test_sample.locator].empty:
            r = df[df["locator_a"] == test_sample.locator].iloc[0]
            pair = "a"
        elif not df[df["locator_b"] == test_sample.locator].empty:
            r = df[df["locator_b"] == test_sample.locator].iloc[0]
            pair = "b"

        bbox = (
            BoundingBox((r[f"{pair}_min_x"], r[f"{pair}_min_y"]), (r[f"{pair}_max_x"], r[f"{pair}_max_y"]))
            if r[f"{pair}_min_x"] is not None and not np.isnan(r[f"{pair}_min_x"])
            else None
        )

        keypoints = (
            Keypoints(
                [
                    (r[f"{pair}_right_eye_x"], r[f"{pair}_right_eye_y"]),
                    (r[f"{pair}_left_eye_x"], r[f"{pair}_left_eye_y"]),
                    (r[f"{pair}_nose_x"], r[f"{pair}_nose_y"]),
                    (r[f"{pair}_right_mouth_x"], r[f"{pair}_right_mouth_y"]),
                    (r[f"{pair}_left_mouth_x"], r[f"{pair}_left_mouth_y"]),
                ],
            )
            if r[f"{pair}_right_eye_x"] is not None and not np.isnan(r[f"{pair}_right_eye_x"])
            else None
        )

        return Inference(similarities=similarities, bbox=bbox, keypoints=keypoints)

    metadata = dict(model=model_name, detector=detector)
    model = Model(f"{model_name} + {detector} [FR]", infer=infer, metadata=metadata)

    configurations = [
        ThresholdConfiguration(false_match_rate=1e-1, iou_threshold=0.5, nmse_threshold=0.5),
    ]

    for test_suite_name in test_suite_names:
        test_suite = TestSuite.load(test_suite_name)
        test(model, test_suite, evaluate_face_recognition_11, configurations, reset=True)


def main(args: Namespace) -> int:
    kolena.initialize(verbose=True)
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
        default=["MTCNN", "HOG"],
        help="Name(s) of detectors(s) used with corresponding model(s).",
    )
    ap.add_argument(
        "--test-suites",
        default=[f"{DATASET} :: gender [FR]", f"{DATASET} :: race [FR]"],
        help="Name(s) of test suite(s) to test.",
    )
    sys.exit(main(ap.parse_args()))
