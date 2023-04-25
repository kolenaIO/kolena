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
