# Copyright 2021-2025 Kolena Inc.
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
from io import StringIO
from typing import Optional
from typing import Set
from unittest.mock import patch

import pytest
from personal_data_detection.utils import detect_pii_in_string


@pytest.mark.parametrize(
    "text, allowed_pii_types, expected_is_pii, expected_printed_message",
    [
        ("random", {}, False, None),
        ("My phone number is 5455-123-4567.", {}, True, "[I-TELEPHONENUM] data detected:  5455-123-4567."),
        ("My name is Obee Nobi.", {}, True, "[I-GIVENNAME] data detected:  Obee"),
        (
            "I live at 432423 Deka St, Tanooti. My phone number is ...",
            {},
            True,
            "[I-BUILDINGNUM] data detected:  432423",
        ),
        ("ninja", {"I-USERNAME"}, False, None),
        ("ninja", {}, True, "[I-USERNAME] data detected: ninja"),
    ],
)
def test__detect_pii_in_string(
    text: str,
    allowed_pii_types: Set[str],
    expected_is_pii: bool,
    expected_printed_message: Optional[str],
) -> None:
    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        is_pii = detect_pii_in_string(text, allowed_pii_types=allowed_pii_types)
        printed_message = mock_stdout.getvalue()

    assert is_pii == expected_is_pii
    if expected_printed_message:
        assert expected_printed_message in printed_message
