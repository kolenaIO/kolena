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
import sys
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

from object_detection_3d.evaluator import KITTI3DConfig
from object_detection_3d.evaluator import KITTI3DEvaluator
from object_detection_3d.evaluator import KITTIDifficulty
from object_detection_3d.seed_test_suite import DEFAULT_TEST_SUITE_NAME
from object_detection_3d.workflow import Inference
from object_detection_3d.workflow import Model
from object_detection_3d.workflow import TestSample
from object_detection_3d.workflow import TestSuite

import kolena
from kolena.workflow import test
from kolena.workflow.annotation import ScoredLabeledBoundingBox
from kolena.workflow.annotation import ScoredLabeledBoundingBox3D


def as_2d_bbox(coords: List[float], score: float, label: str) -> ScoredLabeledBoundingBox:
    return ScoredLabeledBoundingBox(
        label=label,
        score=score,
        top_left=(coords[0], coords[1]),
        bottom_right=(coords[2], coords[3]),
    )


def as_3d_bbox(coords: List[float], score: float, label: str) -> ScoredLabeledBoundingBox3D:
    return ScoredLabeledBoundingBox3D(
        label=label,
        score=score,
        dimensions=(coords[3], coords[4], coords[5]),
        center=(coords[0], coords[1], coords[2] + (coords[5] / 2.0)),
        rotations=(0.0, 0.0, coords[6]),
    )


def as_inference(raw: Dict[str, Any]) -> Inference:
    return Inference(
        bboxes_2d=[as_2d_bbox(box["box"], box["score"], box["pred"]) for box in raw["bboxes"]],
        bboxes_3d=[as_3d_bbox(box["box3d"], box["score"], box["pred"]) for box in raw["bboxes"]],
    )


def load_results(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def seed_test_run(test_suite_name: str, model_name: str, results: List[Dict[str, Any]]) -> None:
    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)

    inference_by_id = {result["label_id"]: result for result in results}

    def infer(test_sample: TestSample) -> Inference:
        label_id = Path(test_sample.locator).stem
        return as_inference(inference_by_id[label_id])

    model = Model(model_name, infer=infer)
    print(f"using model: {model}")

    test_suite = TestSuite.load(test_suite_name)
    print(f"using test suite: {test_suite}")

    evaluator = KITTI3DEvaluator(
        configurations=[
            KITTI3DConfig(KITTIDifficulty.EASY),
            KITTI3DConfig(KITTIDifficulty.MODERATE),
            KITTI3DConfig(KITTIDifficulty.HARD),
        ],
    )
    print(f"using evaluator: {evaluator}")

    test(model, test_suite, evaluator, reset=True)


def main(args: Namespace) -> int:
    results = load_results(str(args.model_results_file))
    seed_test_run(args.test_suite, args.model, results["results"])
    return 0


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("model", help="Model name.")
    ap.add_argument("model_results_file", help="Name of model results file.")
    ap.add_argument("--test-suite", help="Name of test suite to test.", default=DEFAULT_TEST_SUITE_NAME)

    sys.exit(main(ap.parse_args()))
