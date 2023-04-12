from typing import List
from typing import Tuple

from pydantic import validate_arguments
from shapely.geometry import Polygon
from shapely.validation import make_valid

from kolena._utils.validators import ValidatorConfig


def make_valid_polygon(points: List[Tuple[float, float]]) -> None:
    return make_valid(Polygon(points))


@validate_arguments(config=ValidatorConfig)
def validate_polygon(points: List[Tuple[float, float]]) -> None:
    try:
        make_valid_polygon(points)
    except Exception as exception:
        raise ValueError("Point set is an invalid polygon", points) from exception
