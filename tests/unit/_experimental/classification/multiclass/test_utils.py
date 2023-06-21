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
from typing import Optional
from typing import Tuple

import numpy as np
import pytest

from kolena._experimental.classification.multiclass._utils import get_histogram_range
from kolena._experimental.classification.multiclass._utils import roc_curve


def test__roc_curve():
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
        ([1.02, 4.2, 30.0], None),
        ([-10.0, 0.0, 0.8], None),
        ([-0.3, 4.2, 30.0], None),
    ],
)
def test__get_histogram_range(scores: List[float], expected_range: Optional[Tuple[float, float, int]]):
    got = get_histogram_range(scores)
    assert expected_range == got
