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
from argparse import ArgumentParser
from argparse import Namespace
from random import randint

from keypoint_detection.evaluator import KeypointsEvaluator
from keypoint_detection.workflow import Inference
from keypoint_detection.workflow import Model
from keypoint_detection.workflow import NmseThreshold
from keypoint_detection.workflow import TestSample
from keypoint_detection.workflow import TestSuite

import kolena
from kolena.workflow import test
from kolena.workflow.annotation import Keypoints


def infer(test_sample: TestSample) -> Inference:
    """
    1. load the image pointed to at `test_sample.locator`
    2. pass the image to our model and transform its output into an `Inference` object
    """

    # Generate the dummy inference for the demo purpose.
    return Inference(face=Keypoints([(randint(100, 400), randint(100, 400)) for _ in range(5)]))


def run(args: Namespace) -> None:
    model = Model(args.model_name, infer=infer, metadata=dict(description="Any freeform metadata can go here"))

    test_suite = TestSuite(args.test_suite)
    evaluator = KeypointsEvaluator(
        configurations=[
            NmseThreshold(0.01),
            NmseThreshold(0.02),
            NmseThreshold(0.05),
        ],
    )
    test(model, test_suite, evaluator, reset=True)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("model_name", type=str, help="Name of model to test.")
    ap.add_argument("test_suite", type=str, default="none", help="Name of the test suite to run.")
    kolena.initialize(verbose=True)

    run(ap.parse_args())


if __name__ == "__main__":
    main()
