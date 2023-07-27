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
"""
Annotations are visualized in Kolena as overlays on top of [`TestSample`][kolena.workflow.TestSample] objects.

The following annotation types are available:

- [`BoundingBox`][kolena.workflow.annotation.BoundingBox]
- [`Polygon`][kolena.workflow.annotation.Polygon]
- [`Polyline`][kolena.workflow.annotation.Polyline]
- [`Keypoints`][kolena.workflow.annotation.Keypoints]
- [`BoundingBox3D`][kolena.workflow.annotation.BoundingBox3D]
- [`SegmentationMask`][kolena.workflow.annotation.SegmentationMask]
- [`BitmapMask`][kolena.workflow.annotation.BitmapMask]
- [`ClassificationLabel`][kolena.workflow.annotation.ClassificationLabel]

For example, when viewing images in the Studio, any annotations (such as lists of
[`BoundingBox`][kolena.workflow.annotation.BoundingBox] objects) present in the
[`TestSample`][kolena.workflow.TestSample], [`GroundTruth`][kolena.workflow.GroundTruth],
[`Inference`][kolena.workflow.Inference], or [`MetricsTestSample`][kolena.workflow.MetricsTestSample] objects are
rendered on top of the image.
"""
from abc import ABCMeta
from typing import Dict
from typing import List
from typing import Tuple

from pydantic.dataclasses import dataclass

from kolena._utils.validators import ValidatorConfig
from kolena.workflow._datatypes import DataType
from kolena.workflow._datatypes import TypedDataObject


class _AnnotationType(DataType):
    BOUNDING_BOX = "BOUNDING_BOX"
    POLYGON = "POLYGON"
    POLYLINE = "POLYLINE"
    KEYPOINTS = "KEYPOINTS"
    BOUNDING_BOX_3D = "BOUNDING_BOX_3D"
    SEGMENTATION_MASK = "SEGMENTATION_MASK"
    BITMAP_MASK = "BITMAP_MASK"
    CLASSIFICATION_LABEL = "LABEL"

    @staticmethod
    def _data_category() -> str:
        return "ANNOTATION"


@dataclass(frozen=True, config=ValidatorConfig)
class Annotation(TypedDataObject[_AnnotationType], metaclass=ABCMeta):
    """The base class for all annotation types."""


@dataclass(frozen=True, config=ValidatorConfig)
class BoundingBox(Annotation):
    """Rectangular bounding box specified with pixel coordinates of the top left and bottom right vertices."""

    top_left: Tuple[float, float]
    """The top left vertex (in `(x, y)` image coordinates) of this bounding box."""

    bottom_right: Tuple[float, float]
    """The bottom right vertex (in `(x, y)` image coordinates) of this bounding box."""

    @staticmethod
    def _data_type() -> _AnnotationType:
        return _AnnotationType.BOUNDING_BOX


@dataclass(frozen=True, config=ValidatorConfig)
class LabeledBoundingBox(BoundingBox):
    """
    Rectangular bounding box specified with pixel coordinates of the top left and bottom right vertices and a string
    label.
    """

    label: str
    """The label (e.g. model classification) associated with this bounding box."""


@dataclass(frozen=True, config=ValidatorConfig)
class ScoredBoundingBox(BoundingBox):
    """
    Rectangular bounding box specified with pixel coordinates of the top left and bottom right vertices and a float
    score.
    """

    score: float
    """The score (e.g. model confidence) associated with this bounding box."""


@dataclass(frozen=True, config=ValidatorConfig)
class ScoredLabeledBoundingBox(BoundingBox):
    """
    Rectangular bounding box specified with pixel coordinates of the top left and bottom right vertices, a string
    label, and a float score.
    """

    label: str
    """The label (e.g. model classification) associated with this bounding box."""

    score: float
    """The score (e.g. model confidence) associated with this bounding box."""


@dataclass(frozen=True, config=ValidatorConfig)
class Polygon(Annotation):
    """Arbitrary polygon specified by three or more pixel coordinates."""

    points: List[Tuple[float, float]]
    """The sequence of `(x, y)` points comprising the boundary of this polygon."""

    @staticmethod
    def _data_type() -> _AnnotationType:
        return _AnnotationType.POLYGON

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError(f"{type(self).__name__} must have at least three points ({len(self.points)} provided)")


@dataclass(frozen=True, config=ValidatorConfig)
class LabeledPolygon(Polygon):
    """Arbitrary polygon specified by three or more pixel coordinates and a string label."""

    label: str
    """The label (e.g. model classification) associated with this polygon."""


@dataclass(frozen=True, config=ValidatorConfig)
class ScoredPolygon(Polygon):
    """
    Arbitrary polygon specified by three or more pixel coordinates and a float score.
    """

    score: float
    """The score (e.g. model confidence) associated with this polygon."""


@dataclass(frozen=True, config=ValidatorConfig)
class ScoredLabeledPolygon(Polygon):
    """
    Arbitrary polygon specified by three or more pixel coordinates with a string label and a float score.
    """

    label: str
    """The label (e.g. model classification) associated with this polygon."""

    score: float
    """The score (e.g. model confidence) associated with this polygon."""


@dataclass(frozen=True, config=ValidatorConfig)
class Keypoints(Annotation):
    """Array of any number of keypoints specified in pixel coordinates."""

    points: List[Tuple[float, float]]
    """The sequence of discrete `(x, y)` points comprising this keypoints annotation."""

    @staticmethod
    def _data_type() -> _AnnotationType:
        return _AnnotationType.KEYPOINTS


@dataclass(frozen=True, config=ValidatorConfig)
class Polyline(Annotation):
    """Polyline with any number of vertices specified in pixel coordinates."""

    points: List[Tuple[float, float]]
    """The sequence of connected `(x, y)` points comprising this polyline."""

    @staticmethod
    def _data_type() -> _AnnotationType:
        return _AnnotationType.POLYLINE


@dataclass(frozen=True, config=ValidatorConfig)
class BoundingBox3D(Annotation):
    """
    Three-dimensional cuboid bounding box in a right-handed coordinate system.

    Specified by `(x, y, z)` coordinates for the `center` of the cuboid, `(x, y, z)` `dimensions`, and a `rotation`
    parameter specifying the degrees of rotation about each axis `(x, y, z)` ranging `[-π, π]`.
    """

    center: Tuple[float, float, float]
    """`(x, y, z)` coordinates specifying the center of the bounding box."""

    dimensions: Tuple[float, float, float]
    """`(x, y, z)` measurements specifying the dimensions of the bounding box."""

    rotations: Tuple[float, float, float]
    """Rotations in degrees about each `(x, y, z)` axis."""

    @staticmethod
    def _data_type() -> _AnnotationType:
        return _AnnotationType.BOUNDING_BOX_3D


@dataclass(frozen=True, config=ValidatorConfig)
class LabeledBoundingBox3D(BoundingBox3D):
    """[`BoundingBox3D`][kolena.workflow.annotation.BoundingBox3D] with an additional string label."""

    label: str
    """The label associated with this 3D bounding box."""


@dataclass(frozen=True, config=ValidatorConfig)
class ScoredBoundingBox3D(BoundingBox3D):
    """[`BoundingBox3D`][kolena.workflow.annotation.BoundingBox3D] with an additional float score."""

    score: float
    """The score associated with this 3D bounding box."""


@dataclass(frozen=True, config=ValidatorConfig)
class ScoredLabeledBoundingBox3D(BoundingBox3D):
    """[`BoundingBox3D`][kolena.workflow.annotation.BoundingBox3D] with an additional string label and float score."""

    label: str
    """The label associated with this 3D bounding box."""

    score: float
    """The score associated with this 3D bounding box."""


@dataclass(frozen=True, config=ValidatorConfig)
class SegmentationMask(Annotation):
    """
    Raster segmentation mask. The `locator` is the URL to the image file representing the segmentation mask.

    The segmentation mask must be rendered as a single-channel, 8-bit-depth (grayscale) image. For the best results,
    use a lossless file format such as PNG. Each pixel's value is the numerical ID of its class label, as specified in
    the `labels` map. Any pixel value not present in the `labels` map is rendered as part of the background.

    For example, `labels = {255: "object"}` will highlight all pixels with the value of 255 as `"object"`. Every
    other pixel value will be transparent.
    """

    labels: Dict[int, str]
    """Mapping of unique label IDs (pixel values) to unique label values."""

    locator: str
    """URL of the segmentation mask image."""

    @staticmethod
    def _data_type() -> _AnnotationType:
        return _AnnotationType.SEGMENTATION_MASK


@dataclass(frozen=True, config=ValidatorConfig)
class BitmapMask(Annotation):
    """Arbitrary bitmap mask. The `locator` is the URL to the image file representing the mask."""

    locator: str
    """URL of the bitmap data."""

    @staticmethod
    def _data_type() -> _AnnotationType:
        return _AnnotationType.BITMAP_MASK


@dataclass(frozen=True, config=ValidatorConfig)
class ClassificationLabel(Annotation):
    """Label of classification."""

    label: str
    """String label for this classification."""

    @staticmethod
    def _data_type() -> _AnnotationType:
        return _AnnotationType.CLASSIFICATION_LABEL


@dataclass(frozen=True, config=ValidatorConfig)
class ScoredClassificationLabel(ClassificationLabel):
    """Classification label with accompanying score."""

    score: float
    """Score associated with this label."""


_ANNOTATION_TYPES = [
    BoundingBox,
    LabeledBoundingBox,
    ScoredBoundingBox,
    ScoredLabeledBoundingBox,
    Polygon,
    LabeledPolygon,
    ScoredPolygon,
    ScoredLabeledPolygon,
    Keypoints,
    Polyline,
    BoundingBox3D,
    LabeledBoundingBox3D,
    ScoredBoundingBox3D,
    ScoredLabeledBoundingBox3D,
    SegmentationMask,
    BitmapMask,
    ClassificationLabel,
    ScoredClassificationLabel,
]
