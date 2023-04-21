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
from abc import ABCMeta
from typing import Optional

from pydantic.dataclasses import dataclass

from kolena._utils.validators import ValidatorConfig
from kolena.workflow._datatypes import DataType
from kolena.workflow._datatypes import TypedDataObject


class _AssetType(DataType):
    IMAGE = "IMAGE"
    PLAIN_TEXT = "PLAIN_TEXT"
    BINARY = "BINARY"
    POINT_CLOUD = "POINT_CLOUD"
    VIDEO = "VIDEO"

    @staticmethod
    def _data_category() -> str:
        return "ASSET"


@dataclass(frozen=True, config=ValidatorConfig)
class Asset(TypedDataObject[_AssetType], metaclass=ABCMeta):
    """Assets are hyperlinked objects that can be visualized in the web platform when viewing test samples."""


@dataclass(frozen=True, config=ValidatorConfig)
class ImageAsset(Asset):
    """An image in a cloud bucket."""

    locator: str

    @staticmethod
    def _data_type() -> _AssetType:
        return _AssetType.IMAGE


@dataclass(frozen=True, config=ValidatorConfig)
class PlainTextAsset(Asset):
    """A plain text file in a cloud bucket."""

    locator: str

    @staticmethod
    def _data_type() -> _AssetType:
        return _AssetType.PLAIN_TEXT


@dataclass(frozen=True, config=ValidatorConfig)
class BinaryAsset(Asset):
    """A binary file in a cloud bucket."""

    locator: str

    @staticmethod
    def _data_type() -> _AssetType:
        return _AssetType.BINARY


@dataclass(frozen=True, config=ValidatorConfig)
class PointCloudAsset(Asset):
    """
    A three-dimensional point cloud located in a cloud bucket. Points are assumed to be specified in a right-handed,
    Z-up coordinate system with the origin around the sensor that captured the point cloud.
    """

    locator: str

    @staticmethod
    def _data_type() -> _AssetType:
        return _AssetType.POINT_CLOUD


# NOTE: declare BaseVideoAsset as separate class for extension -- default fields in main VideoAsset class prevent
# extension with non-default fields
@dataclass(frozen=True, config=ValidatorConfig)
class BaseVideoAsset(Asset):
    """A video clip located in a cloud bucket or served at a URL."""

    #: URL (e.g. S3, HTTPS) of the video file
    locator: str

    @classmethod
    def _data_type(cls) -> _AssetType:
        return _AssetType.VIDEO


@dataclass(frozen=True, config=ValidatorConfig)
class VideoAsset(BaseVideoAsset):
    """A video clip located in a cloud bucket or served at a URL."""

    #: Optionally provide asset locator for custom video thumbnail
    thumbnail: Optional[ImageAsset] = None

    #: Optionally specify start time of video snippet, in seconds
    start: Optional[float] = None

    #: Optionally specify end time of video snippet, in seconds
    end: Optional[float] = None

    def __post_init__(self) -> None:
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError(f"Specified start time '{self.start}' is after specified end time '{self.end}'")
        if self.start is not None and self.end is not None and (self.start < 0 or self.end < 0):
            raise ValueError(f"Specified start time '{self.start}' and end time '{self.end}' must be non-negative")


_ASSET_TYPES = [ImageAsset, PlainTextAsset, BinaryAsset, PointCloudAsset, BaseVideoAsset, VideoAsset]
