import json
import os
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
    df = pd.read_csv(f"{BUCKET}/{DATASET}/meta/metadata.csv")

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

    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)

    run(ap.parse_args())


if __name__ == "__main__":
    main()
