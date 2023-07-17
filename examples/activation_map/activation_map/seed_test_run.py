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

import pandas as pd
from activation_map.evaluator import evaluate_dummy
from activation_map.workflow import Inference
from activation_map.workflow import Model
from activation_map.workflow import TestSample
from activation_map.workflow import TestSuite

import kolena
from kolena.workflow.annotation import BitmapMask
from kolena.workflow.test_run import test

BUCKET = "kolena-public-datasets"
FOLDER = "advanced-usage/uploading-activation-map/"
META_DIR = "meta"


def main() -> None:
    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)

    metadata_filepath = os.path.join("s3://", BUCKET, FOLDER, META_DIR, "activation_maps.csv")
    df_metadata = pd.read_csv(metadata_filepath).set_index("locator")

    # 1. Load test suite
    test_suite = TestSuite("activation map example")
    print(f"loaded test suite: '{test_suite}'")

    # 2. run test
    def infer(test_sample: TestSample) -> Inference:
        # loading the locator of previously seeded bitmap mask
        return Inference(activation_map=BitmapMask(df_metadata.loc[test_sample.locator, "bitmap_mask_locator"]))

    model = Model("activation map example", infer=infer)
    test(model, test_suite, evaluate_dummy, reset=True)


if __name__ == "__main__":
    main()
