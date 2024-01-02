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
import sys
from argparse import ArgumentParser
from argparse import Namespace
from typing import List

from semantic_segmentation.activation_map_uploader import ActivationMapUploader
from semantic_segmentation.constants import BUCKET
from semantic_segmentation.constants import DATASET
from semantic_segmentation.utils import sanitize_model_name
from semantic_segmentation.workflow import TestSuite

import kolena


def seed_activation_map(model_name: str, test_suite_names: List[str], out_bucket: str) -> None:
    sanitized_model_name = sanitize_model_name(model_name)
    inf_locator_prefix = f"s3://{BUCKET}/{DATASET}/results/{sanitized_model_name}/"
    map_locator_prefix = f"s3://{out_bucket}/{DATASET}/inferences/{sanitized_model_name}/activation/"
    uploader = ActivationMapUploader(inf_locator_prefix, map_locator_prefix)
    test_sample_names = set()

    for test_suite_name in test_suite_names:
        test_suite = TestSuite.load(test_suite_name)
        for _, test_samples in test_suite.load_test_samples():
            test_sample_names.update([ts.metadata["basename"] for ts in test_samples])

    uploader.submit(test_sample_names)


def main(args: Namespace) -> int:
    kolena.initialize(verbose=True)
    seed_activation_map(args.model, args.test_suites, args.out_bucket)
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
        help="Name of AWS S3 bucket with write access to upload activation maps to.",
    )

    sys.exit(main(ap.parse_args()))
