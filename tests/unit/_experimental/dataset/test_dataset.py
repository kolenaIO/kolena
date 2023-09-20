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
import pytest

from kolena._experimental.dataset.dataset import _infer_datatype
from kolena._experimental.dataset.dataset import TEST_SAMPLE_TYPE

CUSTOM = f"{TEST_SAMPLE_TYPE}/CUSTOM"
IMAGE = f"{TEST_SAMPLE_TYPE}/IMAGE"
VIDEO = f"{TEST_SAMPLE_TYPE}/VIDEO"
DOCUMENT = f"{TEST_SAMPLE_TYPE}/DOCUMENT"
TEXT = f"{TEST_SAMPLE_TYPE}/TEXT"
POINT_CLOUD = f"{TEST_SAMPLE_TYPE}/POINT_CLOUD"


@pytest.mark.parametrize(
    "uri,expected",
    [
        ("s3://public/png", CUSTOM),
        ("/opt/test.png", IMAGE),
        ("https://kolena.io/demo.mp4", VIDEO),
        ("file:///var/mime.csv", DOCUMENT),
        ("test.pcd", POINT_CLOUD),
        ("gcp://summary.pdf", DOCUMENT),
        ("//my.mp3", CUSTOM),
    ],
)
def test__infer_datatype(uri: str, expected: str) -> None:
    assert _infer_datatype(uri) == expected


def test__serialize_dataframe() -> None:
    ...


def test__deserialize_dataframe() -> None:
    ...
