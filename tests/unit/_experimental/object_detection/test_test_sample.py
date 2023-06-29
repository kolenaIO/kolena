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
from kolena._experimental.object_detection import TestSample

TEST_LOCATOR = "s3://test-bucket/path/to/file.png"


def test__test__sample__simple() -> None:
    ts = TestSample(
        locator=TEST_LOCATOR,
    )
    assert ts.locator == TEST_LOCATOR
    assert ts.metadata == {}
    assert ts._to_dict() == dict([("locator", "s3://test-bucket/path/to/file.png"), ("data_type", "TEST_SAMPLE/IMAGE")])


def test__test__sample__advanced() -> None:
    metadata = {"name": "mark", "age": 22}

    ts = TestSample(
        locator=TEST_LOCATOR,
        metadata=metadata,
    )
    assert ts.locator == TEST_LOCATOR
    assert ts.metadata == metadata
    assert ts._to_dict() == dict([("locator", "s3://test-bucket/path/to/file.png"), ("data_type", "TEST_SAMPLE/IMAGE")])
