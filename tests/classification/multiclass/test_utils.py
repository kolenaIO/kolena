from typing import List
from typing import Tuple

import numpy as np
import pytest

from kolena.classification.multiclass._utils import get_histogram_range
from kolena.classification.multiclass._utils import roc_curve


def test_roc_curve():
    y_true = [1, 1, 0, 1, 0, 0, 1, 0, 1]
    y_score = [0.3, 0.8, 0.5, 0.7, 0.1, 0.4, 0.9, 0.2, 0.6]
    fpr, tpr = roc_curve(y_true, y_score)
    assert np.allclose(fpr, [0.0, 0.0, 0.0, 0.5, 0.5, 1.0])
    assert np.allclose(tpr, [0.0, 0.2, 0.8, 0.8, 1.0, 1.0])


@pytest.mark.parametrize(
    "scores,expected_range",
    [
        ([0.0, 0.5, 1.0], (0.0, 1.0, 50)),
        ([0.5, 1.0], (0.5, 1.0, 25)),
        ([0.0, 0.79], (0.0, 0.8, 40)),
        ([0.0, 0.79, 0.78], (0.0, 0.8, 40)),
        ([0.45, 0.29, 0.355555], (0.28, 0.46, 9)),
    ],
)
def test_get_histogram_range(scores: List[float], expected_range: Tuple[float, float, int]):
    got = get_histogram_range(scores)
    assert expected_range == got
