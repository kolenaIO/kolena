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
import math
from typing import Optional

import pytest

from kolena._experimental.utils import get_delta_percentage
from kolena._experimental.utils import margin_of_error


@pytest.mark.parametrize(
    "n_samples,confidence_level,expected",
    [
        (50, 0.95, 13.859039),
        (50, 0.9, 11.63087381),
        (500, 0.9, 3.6780052442456856),
        (5, 0.9, 36.78005244245685),
    ],
)
def test__margin_of_error(n_samples: int, confidence_level: float, expected: float) -> None:
    result = margin_of_error(n_samples, confidence_level)
    assert math.isclose(result, expected, rel_tol=1e-4)


@pytest.mark.parametrize(
    "value_a,value_b,max_value,expected",
    [
        (0.12, 0.3, 0.3, -18),
        (0.12, 0.3, None, -149.9999),
        (0.5, 0.3, 1, 20),
        (50, 120, 130, -53.84615384615385),
        (50, 120, None, -140),
        (-0.5, -0.12, None, 76),
        (-0.5, -0.12, -0.1, 76),
    ],
)
def test__get_delta_percentage(value_a: float, value_b: float, max_value: Optional[int], expected: float) -> None:
    result = get_delta_percentage(value_a, value_b, max_value)
    assert math.isclose(result, expected, rel_tol=1e-4)
