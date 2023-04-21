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
from abc import ABCMeta
from abc import abstractmethod
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

from pydantic import validate_arguments

from kolena._utils.frozen import Frozen
from kolena._utils.serializable import Serializable
from kolena._utils.validators import ValidatorConfig
from kolena.errors import InputValidationError


class MetadataElementType(str, Enum):
    BOUNDING_BOX = "BOUNDING_BOX"
    LANDMARKS = "LANDMARKS"  # TODO: migrate terminology to "keypoints", including existing db data.
    ASSET = "ASSET"


class Annotation(Frozen, Serializable, metaclass=ABCMeta):
    """
    An annotation associated with an image.

    Annotations are surfaced during testing along with the image locator and any other metadata associated with an
    image. In the web platform, annotations are overlaid on top of images when visualizing results.
    """

    @abstractmethod
    def __init__(self) -> None:
        ...


class Asset(Frozen, Serializable):
    """
    An asset living in your shared bucket. Assets are surfaced during testing along with any other metadata associated
    with a given test image.

    In the web platform, certain assets such as PNG and JPG images are viewable when visualizing results in the gallery.
    """

    #: Location of this asset in shared bucket, e.g. ``s3://my-bucket/path/to/image.png``.
    locator: str

    @validate_arguments(config=ValidatorConfig)
    def __init__(self, locator: str):
        self.locator = locator  # TODO: validate locator

    def _to_dict(self) -> Dict[str, Any]:
        return dict(data_type=MetadataElementType.ASSET.value, data_object=dict(locator=self.locator))


MetadataScalar = Union[None, str, float, int, bool]
MetadataElement = Union[MetadataScalar, Annotation, Asset]


def _to_dict(metadata: Dict[str, MetadataElement]) -> Dict[str, Any]:
    def dump_element(element: MetadataElement) -> Any:
        if isinstance(element, Serializable):
            return element._to_dict()
        if isinstance(element, float):
            if math.isnan(element):
                return None
            if math.isinf(element):
                raise InputValidationError("cannot serialize infinite metadata values")
        return element

    return {key: dump_element(value) for key, value in metadata.items()}


class BoundingBox(Annotation):
    """An annotation comprising a bounding box around an object in an image."""

    #: Point in (x, y) pixel coordinates representing the top left corner of the bounding box.
    top_left: Tuple[float, float]

    #: Point in (x, y) pixel coordinates representing the bottom right corner of the bounding box.
    bottom_right: Tuple[float, float]

    @validate_arguments(config=ValidatorConfig)
    def __init__(self, top_left: Tuple[float, float], bottom_right: Tuple[float, float]):
        self.top_left = top_left
        self.bottom_right = bottom_right

    def _to_dict(self) -> Dict[str, Any]:
        return dict(
            data_type=MetadataElementType.BOUNDING_BOX.value,
            data_object=dict(points=[self.top_left, self.bottom_right]),
        )


class Landmarks(Annotation):
    """
    An annotation comprising an arbitrary-length set of landmarks corresponding to some object in an image, e.g. face
    landmarks used for pose estimation.
    """

    #: Any number of (x, y) points in pixel coordinates representing a set of landmarks.
    points: List[Tuple[float, float]]

    @validate_arguments(config=ValidatorConfig)
    def __init__(self, points: List[Tuple[float, float]]):
        if len(points) == 0:
            raise ValueError("At least one point required for landmarks annotation.")
        self.points = points

    def _to_dict(self) -> Dict[str, Any]:
        return dict(data_type=MetadataElementType.LANDMARKS.value, data_object=dict(points=self.points))


def _parse_element(element: Union[None, str, float, int, bool, Dict[str, Any]]) -> MetadataElement:
    if element is None or isinstance(element, (str, float, int, bool)):
        return element
    if isinstance(element, dict):
        data_type = element["data_type"]
        data_object = element["data_object"]
        if data_type == MetadataElementType.BOUNDING_BOX.value:
            top_left, bottom_right = data_object["points"]
            return BoundingBox(top_left, bottom_right)
        if data_type == MetadataElementType.LANDMARKS.value:
            return Landmarks(data_object["points"])
        if data_type == MetadataElementType.ASSET.value:
            return Asset(data_object["locator"])
    raise ValueError(f"unrecognized value: {element}")


def _from_dict(metadata_blob: Dict[str, Any]) -> Dict[str, MetadataElement]:
    return {key: _parse_element(value) for key, value in metadata_blob.items()}
