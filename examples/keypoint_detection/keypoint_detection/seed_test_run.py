import os
from argparse import ArgumentParser
from argparse import Namespace
from random import randint
from random import random

from keypoint_detection.evaluator import KeypointsEvaluator
from keypoint_detection.workflow import Inference
from keypoint_detection.workflow import Model
from keypoint_detection.workflow import NmeThreshold
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
    return Inference(Keypoints([(randint(100, 400), randint(100, 400)) for _ in range(5)]), random())


def run(args: Namespace) -> None:
    model = Model(args.model_name, infer=infer, metadata=dict(description="Any freeform metadata can go here"))

    test_suite = TestSuite(args.test_suite)
    evaluator = KeypointsEvaluator(
        configurations=[
            NmeThreshold(0.01),
            NmeThreshold(0.05),
            NmeThreshold(0.1),
        ],
    )
    test(model, test_suite, evaluator)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("model_name", type=str, help="Name of model in directory to test.")
    ap.add_argument("test_suite", type=str, default="none", help="Name of the test suite to run.")
    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)

    run(ap.parse_args())


if __name__ == "__main__":
    main()
