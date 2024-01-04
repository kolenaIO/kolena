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
import dataclasses
from typing import Callable
from typing import Type

import pydantic
import pytest

from kolena._utils.datatypes import DATA_TYPE_FIELD
from kolena.workflow.asset import _AssetType
from kolena.workflow.asset import Asset
from kolena.workflow.asset import BinaryAsset
from kolena.workflow.asset import ImageAsset
from kolena.workflow.asset import PlainTextAsset
from kolena.workflow.asset import PointCloudAsset
from kolena.workflow.asset import VideoAsset


@pytest.mark.parametrize(
    "asset_class,asset_type",
    [
        (ImageAsset, _AssetType.IMAGE),
        (PointCloudAsset, _AssetType.POINT_CLOUD),
        (PlainTextAsset, _AssetType.PLAIN_TEXT),
        (BinaryAsset, _AssetType.BINARY),
    ],
)
def test__serialize__locator(asset_class: Type[Asset], asset_type: _AssetType) -> None:
    locator = "s3://fake/locator.jpg"
    asset = asset_class(locator=locator)  # type: ignore
    asset_dict = asset._to_dict()

    assert asset_dict == {"locator": locator, DATA_TYPE_FIELD: f"{_AssetType._data_category()}/{asset_type.value}"}
    assert asset == asset_class._from_dict(asset_dict)


def test__serialize__video() -> None:
    locator = "s3://fake/locator.jpg"
    asset = VideoAsset(locator=locator, start=0, end=1)  # type: ignore
    asset_dict = asset._to_dict()

    assert asset_dict == {
        "locator": locator,
        DATA_TYPE_FIELD: f"{_AssetType.VIDEO._data_category()}/{_AssetType.VIDEO.value}",
        "thumbnail": None,
        "start": 0,
        "end": 1,
    }
    assert asset == VideoAsset._from_dict(asset_dict)


@pytest.mark.parametrize("dataclass", [dataclasses.dataclass, pydantic.dataclasses.dataclass])
def test__instantiate__video(dataclass: Callable) -> None:
    @dataclass(frozen=True)
    class Tester(VideoAsset):
        a: int = 0

    locator = "s3://test-bucket/video.mp4"
    thumbnail = ImageAsset(locator="https://example.com/test.png")
    Tester(locator=locator, thumbnail=thumbnail, start=0, end=1, a=1)
    Tester(locator=locator)
    Tester(locator=locator, start=1, end=1.5)
    Tester(locator=locator, a=1)

    with pytest.raises(ValueError):
        Tester(locator=locator, start=1, end=0)

    with pytest.raises(ValueError):
        Tester(locator=locator, start=-1, end=0)

    with pytest.raises(ValueError):
        Tester(locator=locator, start=-2, end=-1)
