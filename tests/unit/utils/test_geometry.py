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
from typing import List
from typing import Tuple

import pytest

from kolena._utils.geometry import validate_polygon


@pytest.mark.parametrize(
    "points, is_valid",
    [
        ([(0, 0), (0, 0)], False),
        ([(0, 0), 0, 1], False),
        ([(0, 0), (0, 0), (0, 0)], True),
        ([(0, 0), (1, 0), (1, 1)], True),
        ([(0, 0), (0.5, 0.5), (1, 0), (1, 1)], True),  # co-linear points
    ],
)
def test_validate_polygon(points: List[Tuple[float, float]], is_valid: bool) -> None:
    if not is_valid:
        with pytest.raises(ValueError):
            validate_polygon(points)
    else:
        validate_polygon(points)
