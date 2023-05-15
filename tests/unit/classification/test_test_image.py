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
import kolena.classification.metadata
from kolena.classification import TestImage


def test__test_image__serde() -> None:
    original = TestImage(
        locator="s3://test-bucket/path/to/file.png",
        dataset="test-dataset",
        labels=["one", "2", "$3", "^4", "!@#$%^&*()"],
        metadata={
            "none": None,
            "str": "string",
            "float": 1.0,
            "int": 500,
            "bool": False,
            "bbox": kolena.classification.metadata.BoundingBox(top_left=(0, 0), bottom_right=(100, 100)),
            "landmarks": kolena.classification.metadata.Landmarks(points=[(0, 0), (100, 100), (100, 0), (0, 0)]),
            "asset": kolena.classification.metadata.Asset(locator="gs://gs-bucket/path/to/asset.jpg"),
        },
    )

    df = kolena.classification.test_case.TestCase._to_data_frame([original])
    assert len(df) == 1

    recovered = [TestImage._from_record(record) for record in df.itertuples()][0]
    assert original == recovered
