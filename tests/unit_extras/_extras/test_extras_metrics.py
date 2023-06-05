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

import numpy as np
import pytest
from kolena._extras.metrics import sklearn_metrics

@pytest.mark.parametrize(
    "y_true, y_pred, precision, recall",
    [
        (
            np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig']),
            np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog']),
            2/9,
            1/3,
        ),
    ],
)
def test__extras__sklearn__pr(y_true: np.ndarray, y_pred: np.ndarray, precision: float, recall: float) -> None:
    prec, rec, _, _ = sklearn_metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')
    assert precision == pytest.approx(prec, abs=1e-5)
    assert recall == pytest.approx(rec, abs=1e-5)


