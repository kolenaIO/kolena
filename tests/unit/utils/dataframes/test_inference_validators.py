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
import pytest

from kolena._utils.inference_validators import validate_confidence
from kolena._utils.inference_validators import validate_label


@pytest.mark.parametrize(
    "label, is_valid",
    [
        ("label", True),
        ("label with spaces", True),
        ("", False),
        ("   ", False),
        (" \n\r  ", False),
    ],
)
def test__validate_label(label: str, is_valid: bool) -> None:
    if not is_valid:
        with pytest.raises(ValueError):
            validate_label(label)
    else:
        validate_label(label)


@pytest.mark.parametrize(
    "confidence, is_valid",
    [
        (0, True),
        (1, True),
        (1 / 3, True),
        (0.987654321, True),
        (-0, True),
        (-1, False),
        (2, False),
        (None, False),
        (float("nan"), False),
        (float("inf"), False),
        (-float("inf"), False),
    ],
)
def test__validate_confidence(confidence: float, is_valid: bool) -> None:
    if not is_valid:
        with pytest.raises(ValueError):
            validate_confidence(confidence)
    else:
        validate_confidence(confidence)
