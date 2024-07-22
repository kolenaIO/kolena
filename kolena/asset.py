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
"""
Assets are additional files that can be linked to the datapoints in your datasets.
Assets can be visualized in the Kolena when exploring your datasets, test cases, or model results.

The following asset types are available:

- [`ImageAsset`][kolena.asset.ImageAsset]
- [`PlainTextAsset`][kolena.asset.PlainTextAsset]
- [`BinaryAsset`][kolena.asset.BinaryAsset]
- [`PointCloudAsset`][kolena.asset.PointCloudAsset]
- [`VideoAsset`][kolena.asset.VideoAsset]
- [`AudioAsset`][kolena.asset.AudioAsset]

"""
from abc import ABCMeta
from typing import Optional

from kolena._utils.datatypes import DataCategory
from kolena._utils.datatypes import DataType
from kolena._utils.datatypes import TypedDataObject
from kolena._utils.pydantic_v1.dataclasses import dataclass
from kolena._utils.validators import ValidatorConfig


class _AssetType(DataType):
    IMAGE = "IMAGE"
    PLAIN_TEXT = "PLAIN_TEXT"
    BINARY = "BINARY"
    POINT_CLOUD = "POINT_CLOUD"
    VIDEO = "VIDEO"
    AUDIO = "AUDIO"

    @staticmethod
    def _data_category() -> DataCategory:
        return DataCategory.ASSET


@dataclass(frozen=True, config=ValidatorConfig)
class Asset(TypedDataObject[_AssetType], metaclass=ABCMeta):
    """Base class for all asset types."""


@dataclass(frozen=True, config=ValidatorConfig)
class ImageAsset(Asset):
    """An image in a cloud bucket."""

    locator: str
    """The location of this image in a cloud bucket, e.g. `s3://my-bucket/path/to/my-image-asset.png`."""

    @staticmethod
    def _data_type() -> _AssetType:
        return _AssetType.IMAGE


@dataclass(frozen=True, config=ValidatorConfig)
class PlainTextAsset(Asset):
    """A plain text file in a cloud bucket."""

    locator: str
    """The location of this text file in a cloud bucket, e.g. `s3://my-bucket/path/to/my-text-asset.txt`."""

    @staticmethod
    def _data_type() -> _AssetType:
        return _AssetType.PLAIN_TEXT


@dataclass(frozen=True, config=ValidatorConfig)
class BinaryAsset(Asset):
    """A binary file in a cloud bucket."""

    locator: str
    """The location of this text file in a cloud bucket, e.g. `s3://my-bucket/path/to/my-binary-asset.bin`."""

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
    """The location of this point cloud in a cloud bucket, e.g. `s3://my-bucket/path/to/my-point-cloud.pcd`."""

    @staticmethod
    def _data_type() -> _AssetType:
        return _AssetType.POINT_CLOUD


# NOTE: declare BaseVideoAsset as separate class for extension -- default fields in main VideoAsset class prevent
# extension with non-default fields
@dataclass(frozen=True, config=ValidatorConfig)
class BaseVideoAsset(Asset):
    """A video clip located in a cloud bucket or served at a URL."""

    locator: str
    """URL (e.g. S3, HTTPS) of the video file."""

    @classmethod
    def _data_type(cls) -> _AssetType:
        return _AssetType.VIDEO


@dataclass(frozen=True, config=ValidatorConfig)
class VideoAsset(BaseVideoAsset):
    """A video clip located in a cloud bucket or served at a URL."""

    locator: str
    """URL (e.g. S3, HTTPS) of the video file."""

    thumbnail: Optional[ImageAsset] = None
    """Optionally provide asset locator for custom video thumbnail image."""

    start: Optional[float] = None
    """Optionally specify start time of video snippet, in seconds."""

    end: Optional[float] = None
    """Optionally specify end time of video snippet, in seconds."""

    def __post_init__(self) -> None:
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError(f"Specified start time '{self.start}' is after specified end time '{self.end}'")
        if self.start is not None and self.end is not None and (self.start < 0 or self.end < 0):
            raise ValueError(f"Specified start time '{self.start}' and end time '{self.end}' must be non-negative")


@dataclass(frozen=True, config=ValidatorConfig)
class AudioAsset(Asset):
    """
    An audio file in a cloud bucket or served at a URL.

    For best results, use a broadly supported file type such as `.mp3` or `.wav`.
    """

    locator: str
    """The location of this audio file in a cloud bucket, e.g. `s3://my-bucket/path/to/my-audio-asset.mp3`."""

    @staticmethod
    def _data_type() -> _AssetType:
        return _AssetType.AUDIO


_ASSET_TYPES = [ImageAsset, PlainTextAsset, BinaryAsset, PointCloudAsset, BaseVideoAsset, VideoAsset, AudioAsset]
