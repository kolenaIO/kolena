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
from activation_map.workflow import GroundTruth
from activation_map.workflow import TestCase
from activation_map.workflow import TestSample
from activation_map.workflow import TestSuite

import kolena

BUCKET = "kolena-public-datasets"
FOLDER = "advanced-usage/uploading-activation-map/"
META_DIR = "meta"


def main() -> None:
    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)

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

    test_case = TestCase("activation map example", test_samples=test_samples_and_ground_truths, reset=True)

    test_suite = TestSuite("activation map example", test_cases=[test_case], reset=True)
    print(f"created test suite: '{test_suite}'")


if __name__ == "__main__":
    main()
