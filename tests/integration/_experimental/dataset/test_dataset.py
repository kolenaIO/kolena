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
import math
import random

import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from kolena._experimental.dataset import fetch_dataset
from kolena._experimental.dataset import register_dataset
from kolena.errors import NotFoundError
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import Polyline
from tests.integration.helper import fake_locator
from tests.integration.helper import with_test_prefix


def test__register_dataset__empty() -> None:
    name = with_test_prefix(f"{__file__}::test__register_dataset__empty")
    register_dataset(name, pd.DataFrame())

    assert fetch_dataset(name).empty


def test__register_dataset() -> None:
    name = with_test_prefix(f"{__file__}::test__register_dataset")
    datapoints_p1 = [
        dict(
            locator=fake_locator(i, name),
            width=i + 500,
            height=i + 400,
            city=random.choice(["new york", "waterloo"]),
            bboxes=[
                LabeledBoundingBox(label="cat", top_left=[i, i], bottom_right=[i + 10, i + 10]),
                LabeledBoundingBox(label="dog", top_left=[i + 5, i + 5], bottom_right=[i + 20, i + 20]),
            ],
        )
        for i in range(10)
    ]
    datapoints_p2 = [
        dict(
            locator=fake_locator(i, name),
            location=random.choice(["new york", "waterloo"]),
            polylines=[Polyline(points=[(1, 1), (2, 2), (3, 3)])],
            bboxes=[
                LabeledBoundingBox(label="cat", top_left=[i, i], bottom_right=[i + 10, i + 10]),
                LabeledBoundingBox(label="dog", top_left=[i + 5, i + 5], bottom_right=[i + 20, i + 20]),
            ],
            width=i + 500,
        )
        for i in range(10, 20)
    ]
    datapoints = datapoints_p1 + datapoints_p2
    expected_datapoints = [
        dict(
            locator=dp["locator"],
            width=dp["width"],
            height=dp.get("height", math.nan),
            city=dp.get("city", None),
            bboxes=[
                BoundingBox(label=bbox.label, top_left=bbox.top_left, bottom_right=bbox.bottom_right)
                for bbox in dp["bboxes"]
            ],
            location=dp.get("location", None),
            polylines=dp.get("polylines", None),
        )
        for dp in datapoints
    ]
    expected_datapoints_p1 = [
        dict(
            locator=dp["locator"],
            width=dp["width"],
            height=dp["height"],
            city=dp["city"],
            bboxes=[
                BoundingBox(label=bbox.label, top_left=bbox.top_left, bottom_right=bbox.bottom_right)
                for bbox in dp["bboxes"]
            ],
        )
        for dp in datapoints_p1
    ]
    register_dataset(name, pd.DataFrame(datapoints_p1))

    loaded_datapoints = fetch_dataset(name).sort_values("width", ignore_index=True)
    expected = pd.DataFrame(expected_datapoints_p1)
    assert_frame_equal(loaded_datapoints, expected)

    # update dataset with datapoints of different structure
    datapoints_updated = pd.DataFrame(datapoints[:5] + datapoints[7:15])
    register_dataset(name, datapoints_updated)

    loaded_datapoints = fetch_dataset(name).sort_values("width", ignore_index=True)
    expected = pd.DataFrame(expected_datapoints[:5] + expected_datapoints[7:15])
    assert_frame_equal(loaded_datapoints, expected)


def test__fetch_dataset__not_exist() -> None:
    name = with_test_prefix(f"{__file__}::test__fetch_dataset__not_exist")
    with pytest.raises(NotFoundError):
        fetch_dataset(name)
