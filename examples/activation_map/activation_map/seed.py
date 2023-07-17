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
from typing import Dict

import pandas as pd
from activation_map.evaluator import evaluate
from activation_map.utils import bitmap_locator
from activation_map.utils import create_and_upload_bitmap
from activation_map.utils import load_activation_map
from activation_map.workflow import GroundTruth
from activation_map.workflow import Inference
from activation_map.workflow import Model
from activation_map.workflow import TestCase
from activation_map.workflow import TestSample
from activation_map.workflow import TestSuite

import kolena
from kolena.workflow.annotation import BitmapMask
from kolena.workflow.test_run import test

BUCKET = "kolena-public-datasets"
FOLDER = "advanced-usage/uploading-activation-map/"
META_DIR = "meta"


def main(args: Namespace) -> int:
    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)

    # 1. seed a test suite
    metadata_filepath = os.path.join("s3://", BUCKET, FOLDER, META_DIR, "activation_maps.csv")
    df_metadata = pd.read_csv(metadata_filepath)

    test_samples_and_ground_truths = [
        (
            TestSample(
                locator=record.locator,
            ),
            GroundTruth(),
        )
        for record in df_metadata.itertuples(index=False)
    ]
    print(type(test_samples_and_ground_truths[0][1]))

    test_case = TestCase("activation map example", test_samples=test_samples_and_ground_truths)

    test_suite = TestSuite("activation map example", test_cases=[test_case])
    print(f"created test suite: {test_suite}")

    # 2. load and process activation map
    bitmap_mask_by_locator: Dict[str, BitmapMask] = {}

    for record in df_metadata.itertuples(index=False):
        filename = os.path.basename(record.activation_map_locator)
        activation_map = load_activation_map(record.activation_map_locator)
        bitmap_mask_by_locator[record.locator] = create_and_upload_bitmap(
            bitmap_locator(args.write_bucket, args.write_prefix, filename),
            activation_map,
        )

    # 3. run test
    def infer(test_sample: TestSample) -> Inference:
        return Inference(activation_map=bitmap_mask_by_locator[test_sample.locator])

    model = Model("activation map example", infer=infer)
    test(model, test_suite, evaluate)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--write-bucket",
        help="A S3 bucket name where the bitmaps are going to be stored. Ensure to use the AWS credential with "
        "a write access to this bucket.",
    )
    ap.add_argument(
        "--write-prefix",
        help='A subdirectory prefix on S3 bucket where the bitmaps are going to be uploaded (e.g., "example/bitmaps", '
        'then the bitmap of activation map file "filename.npy" is going to be stored under '
        '"s3://{args.upload_s3_bucket}/example/bitmaps/image.png")',
    )
    sys.exit(main(ap.parse_args()))
