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
from kolena._utils.dataframes.validators import _is_locator_cell_valid


def test_validate_locator() -> None:
    valid_locators = [
        "s3://bucket-name/path/to/image.jpg",
        "gs://bucket/path/to/image.png",
        "s3://bucket/image with spaces.jpg",  # spaces should be allowed
        "s3://bucket/UPPERCASE.JPG",  # uppercase
        "gs://bucket/lower.jpG",
        "http://bucket/lower.jpG",
        "https://bucket/lower.jpG",
    ]
    invalid_locators = [
        "garbage",
        "closer://but/still/garbage.jpg",
        "s3://image.jpg",  # missing bucket name
        "s3://bucket/image.txt",  # non-image extension
        "s3:/bucket/image.jpg",  # malformed
    ]

    for locator in valid_locators:
        assert _is_locator_cell_valid(locator)

    for locator in invalid_locators:
        assert not _is_locator_cell_valid(locator)
