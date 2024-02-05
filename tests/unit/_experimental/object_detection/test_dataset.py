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
import pandas as pd

from kolena._experimental.object_detection.dataset import _check_multiclass
from kolena.annotation import LabeledBoundingBox


def test__check_multiclass() -> None:
    assert (
        _check_multiclass(
            pd.Series(
                [
                    [LabeledBoundingBox(label="dog", top_left=[1, 1], bottom_right=[5, 5])],
                    [LabeledBoundingBox(label="dog", top_left=[10, 10], bottom_right=[15, 15])],
                ],
            ),
            pd.Series(
                [
                    [LabeledBoundingBox(label="cat", top_left=[3, 3], bottom_right=[9, 9])],
                    [LabeledBoundingBox(label="dog", top_left=[11, 10], bottom_right=[15, 15])],
                ],
            ),
        )
        is True
    )

    assert (
        _check_multiclass(
            pd.Series(
                [
                    [LabeledBoundingBox(label="dog", top_left=[1, 1], bottom_right=[5, 5])],
                    [LabeledBoundingBox(label="dog", top_left=[10, 10], bottom_right=[15, 15])],
                ],
            ),
            pd.Series(
                [
                    [LabeledBoundingBox(label="dog", top_left=[3, 3], bottom_right=[9, 9])],
                    [LabeledBoundingBox(label="dog", top_left=[11, 10], bottom_right=[15, 15])],
                ],
            ),
        )
        is False
    )
