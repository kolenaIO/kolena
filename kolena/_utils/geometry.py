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
from typing import List
from typing import Tuple

from shapely.geometry import Polygon
from shapely.validation import make_valid

from kolena._utils.pydantic_v1 import validate_arguments
from kolena._utils.validators import ValidatorConfig


def make_valid_polygon(points: List[Tuple[float, float]]) -> None:
    return make_valid(Polygon(points))


@validate_arguments(config=ValidatorConfig)
def validate_polygon(points: List[Tuple[float, float]]) -> None:
    try:
        make_valid_polygon(points)
    except Exception as exception:
        raise ValueError("Point set is an invalid polygon", points) from exception
