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
import json

import pandas as pd

import kolena
from kolena.dataset import upload_dataset
from kolena.workflow.annotation import Keypoints

DATASET = "300-W"
BUCKET = "kolena-public-examples"


def main() -> None:
    df = pd.read_csv(f"s3://{BUCKET}/{DATASET}/raw/{DATASET}.csv", index_col=0, storage_options={"anon": True})
    df["face"] = df["points"].apply(lambda points: Keypoints(points=json.loads(points)))
    df["condition"] = df["locator"].apply(lambda locator: "indoor" if "indoor" in locator else "outdoor")

    kolena.initialize(verbose=True)
    upload_dataset(DATASET, df[["locator", "face", "normalization_factor", "condition"]])


if __name__ == "__main__":
    main()
