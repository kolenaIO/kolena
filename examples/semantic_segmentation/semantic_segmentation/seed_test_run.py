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

from semantic_segmentation.constants import BUCKET
from semantic_segmentation.constants import DATASET
from semantic_segmentation.evaluator import evaluate_semantic_segmentation
from semantic_segmentation.utils import sanitize_model_name
from semantic_segmentation.workflow import Inference
from semantic_segmentation.workflow import Model
from semantic_segmentation.workflow import SegmentationConfiguration
from semantic_segmentation.workflow import TestSample
from semantic_segmentation.workflow import TestSuite

import kolena
from kolena.workflow.asset import BinaryAsset
from kolena.workflow.test_run import test


def seed_test_run(model_name: str, test_suite_names: List[str]) -> None:
    def infer(test_sample: TestSample) -> Inference:
        basename = test_sample.metadata["basename"]
        locator = f"s3://{BUCKET}/{DATASET}/results/{sanitized_model_name}/{basename}_person.npy"
        return Inference(prob=BinaryAsset(locator))

    sanitized_model_name = sanitize_model_name(model_name)
    model = Model(f"{model_name}", infer=infer)

    for test_suite_name in test_suite_names:
        test_suite = TestSuite.load(test_suite_name)
        configurations = [SegmentationConfiguration(threshold=0.5)]

        test(
            model,
            test_suite,
            evaluate_semantic_segmentation,
            configurations=configurations,
            reset=True,
        )


def main(args: Namespace) -> int:
    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)
    os.environ["KOLENA_MODEL_NAME"] = str(args.model)
    os.environ["KOLENA_OUT_BUCKET"] = str(args.out_bucket)
    seed_test_run(args.model, args.test_suites)
    return 0


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--model",
        default="pspnet_r101-d8_4xb4-40k_coco-stuff10k-512x512",
        help="Name of model in directory to test",
    )
    ap.add_argument(
        "--test_suites",
        default=[f"# of people :: {DATASET} [person]"],
        nargs="+",
        help="Name(s) of test suite(s) to test.",
    )
    ap.add_argument(
        "--out_bucket",
        required=True,
        help="Name of AWS S3 bucket with write access to upload result masks to.",
    )

    sys.exit(main(ap.parse_args()))
