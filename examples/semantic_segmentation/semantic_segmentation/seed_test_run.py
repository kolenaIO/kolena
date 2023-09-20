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

from semantic_segmentation.evaluator import evaluate_semantic_segmentation
from semantic_segmentation.workflow import Inference
from semantic_segmentation.workflow import Model
from semantic_segmentation.workflow import SemanticSegmentationConfiguration
from semantic_segmentation.workflow import TestSample
from semantic_segmentation.workflow import TestSuite

import kolena
from kolena.workflow.asset import BinaryAsset
from kolena.workflow.test_run import test


BUCKET = "kolena-public-datasets"
DATASET = "coco-stuff-10k"


def seed_test_run(model_name: str, test_suite_names: List[str]) -> None:
    def infer(test_sample: TestSample) -> Inference:
        basename = test_sample.metadata["basename"]
        locator = f"s3://{BUCKET}/{DATASET}/results/{model_name}/{basename}_person.npy"
        return Inference(prob=BinaryAsset(locator))

    model = Model(f"{model_name}", infer=infer)
    print(f"Model: {model}")

    for test_suite_name in test_suite_names:
        test_suite = TestSuite.load(test_suite_name)
        print(f"Test Suite: {test_suite}")
        configurations = [SemanticSegmentationConfiguration(model_name=model_name)]

        test(
            model,
            test_suite,
            evaluate_semantic_segmentation,
            configurations,
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
        default=["pspnet_r101-d8_4xb4-40k_coco-stuff10k-512x512"],
        nargs="+",
        help="Name(s) of model(s) in directory to test",
    )
    ap.add_argument(
        "--test_suites",
        default=[f"{DATASET}"],
        nargs="+",
        help="Name(s) of test suite(s) to test.",
    )
    sys.exit(main(ap.parse_args()))
