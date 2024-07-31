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
Annotations are visualized in Kolena as overlays on top of datapoints.

| Annotation | Conditions for visualization |
| --- | --- |
| [`BoundingBox`][kolena.annotation.BoundingBox] | Limited to `Image` or `Video` data |
| [`BoundingBox3D`][kolena.annotation.BoundingBox3D] | Limited to `PointCloud` data |
| [`Polygon`][kolena.annotation.Polygon] | Limited to `Image` or `Video` data |
| [`Polyline`][kolena.annotation.Polyline] | Limited to `Image` or `Video` data |
| [`Keypoints`][kolena.annotation.Keypoints] | Limited to `Image` or `Video` data |
| [`SegmentationMask`][kolena.annotation.SegmentationMask] | Limited to `Image` or `Video` data |
| [`BitmapMask`][kolena.annotation.BitmapMask] | Limited to `Image` or `Video` data |
| [`Label`][kolena.annotation.Label] | Valid for all data |
| [`TimeSegment`][kolena.annotation.TimeSegment] | Limited to `Audio` or `Video` data |
| [`TextSegment`][kolena.annotation.TextSegment] | Limited to `Text` data |

For example, when viewing images in the Studio, any annotations (such as lists of
[`BoundingBox`][kolena.annotation.BoundingBox] objects) present in the datapoints are
rendered on top of the image.
"""  # noqa: E501
import dataclasses
from abc import ABCMeta
from functools import reduce
from typing import Dict
from typing import List
from typing import Tuple

from kolena._utils.datatypes import DataCategory
from kolena._utils.datatypes import DataType
from kolena._utils.datatypes import TypedDataObject
from kolena._utils.pydantic_v1.dataclasses import dataclass
from kolena._utils.validators import ValidatorConfig


class _AnnotationType(DataType):
    BOUNDING_BOX = "BOUNDING_BOX"
    POLYGON = "POLYGON"
    POLYLINE = "POLYLINE"
    POLYLINE_3D = "POLYLINE_3D"
    KEYPOINTS = "KEYPOINTS"
    KEYPOINTS_3D = "KEYPOINTS_3D"
    BOUNDING_BOX_3D = "BOUNDING_BOX_3D"
    SEGMENTATION_MASK = "SEGMENTATION_MASK"
    BITMAP_MASK = "BITMAP_MASK"
    LABEL = "LABEL"
    TIME_SEGMENT = "TIME_SEGMENT"
    TEXT_SEGMENT = "TEXT_SEGMENT"

    @staticmethod
    def _data_category() -> DataCategory:
        return DataCategory.ANNOTATION


@dataclass(frozen=True, config=ValidatorConfig)
class Annotation(TypedDataObject[_AnnotationType], metaclass=ABCMeta):
    """The base class for all annotation types."""


@dataclass(frozen=True, config=ValidatorConfig)
class BoundingBox(Annotation):
    """
    Rectangular bounding box specified with pixel coordinates of the top left and bottom right vertices.

    The reserved fields `width`, `height`, `area`, and `aspect_ratio` are automatically populated with values derived
    from the provided coordinates.
    """

    top_left: Tuple[float, float]
    """The top left vertex (in `(x, y)` pixel coordinates) of this bounding box."""

    bottom_right: Tuple[float, float]
    """The bottom right vertex (in `(x, y)` pixel coordinates) of this bounding box."""

    width: float = dataclasses.field(init=False)
    height: float = dataclasses.field(init=False)
    area: float = dataclasses.field(init=False)
    aspect_ratio: float = dataclasses.field(init=False)

    @staticmethod
    def _data_type() -> _AnnotationType:
        return _AnnotationType.BOUNDING_BOX

    def __post_init__(self) -> None:
        object.__setattr__(self, "width", abs(self.bottom_right[0] - self.top_left[0]))
        object.__setattr__(self, "height", abs(self.bottom_right[1] - self.top_left[1]))
        object.__setattr__(self, "area", self.width * self.height)
        object.__setattr__(self, "aspect_ratio", self.width / self.height if self.height != 0 else 0)


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
    """The sequence of `(x, y)` pixel coordinates comprising the boundary of this polygon."""

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
    """The sequence of discrete `(x, y)` pixel coordinates comprising this keypoints annotation."""

    @staticmethod
    def _data_type() -> _AnnotationType:
        return _AnnotationType.KEYPOINTS


@dataclass(frozen=True, config=ValidatorConfig)
class Keypoints3D(Annotation):
    """Array of any number of keypoints specified in a right-handed coordinate system."""

    points: List[Tuple[float, float, float]]
    """The sequence of discrete `(x, y, z)` coordinates comprising this keypoints annotation."""

    @staticmethod
    def _data_type() -> _AnnotationType:
        return _AnnotationType.KEYPOINTS_3D


@dataclass(frozen=True, config=ValidatorConfig)
class Polyline(Annotation):
    """Polyline with any number of vertices specified in pixel coordinates."""

    points: List[Tuple[float, float]]
    """The sequence of connected `(x, y)` pixel coordinates comprising this polyline."""

    @staticmethod
    def _data_type() -> _AnnotationType:
        return _AnnotationType.POLYLINE


@dataclass(frozen=True, config=ValidatorConfig)
class Polyline3D(Annotation):
    """A three-dimensional Polyline with any number of vertices specified a right-handed coordinate system."""

    points: List[Tuple[float, float, float]]
    """The sequence of connected `(x, y, z)` coordinates comprising this polyline."""

    @staticmethod
    def _data_type() -> _AnnotationType:
        return _AnnotationType.POLYLINE_3D


@dataclass(frozen=True, config=ValidatorConfig)
class BoundingBox3D(Annotation):
    """
    Three-dimensional cuboid bounding box in a right-handed coordinate system.

    Specified by `(x, y, z)` coordinates for the `center` of the cuboid, `(x, y, z)` `dimensions`, and a `rotation`
    parameter specifying the degrees of rotation about each axis `(x, y, z)` ranging `[-π, π]`.

    The reserved field `volume` is automatically derived from the provided `dimensions`.
    """

    center: Tuple[float, float, float]
    """`(x, y, z)` coordinates specifying the center of the bounding box."""

    dimensions: Tuple[float, float, float]
    """`(x, y, z)` measurements specifying the dimensions of the bounding box."""

    rotations: Tuple[float, float, float]
    """Rotations in degrees about each `(x, y, z)` axis."""

    volume: float = dataclasses.field(init=False)

    @staticmethod
    def _data_type() -> _AnnotationType:
        return _AnnotationType.BOUNDING_BOX_3D

    def __post_init__(self) -> None:
        object.__setattr__(self, "volume", reduce(lambda a, b: a * b, self.dimensions))


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
class Label(Annotation):
    """Label, e.g. for classification."""

    label: str
    """String label for this classification."""

    @staticmethod
    def _data_type() -> _AnnotationType:
        return _AnnotationType.LABEL


ClassificationLabel = Label
"""Alias for [`Label`][kolena.workflow.annotation.Label]."""


@dataclass(frozen=True, config=ValidatorConfig)
class ScoredLabel(Label):
    """Label with accompanying score."""

    score: float
    """Score associated with this label."""


ScoredClassificationLabel = ScoredLabel
"""Alias for [`ScoredLabel`][kolena.workflow.annotation.ScoredLabel]."""


@dataclass(frozen=True, config=ValidatorConfig)
class TimeSegment(Annotation):
    """
    Segment of time in the associated audio or video file.

    When a `group` is specified, segments are displayed on Kolena with different colors for each group present in a
    `List[TimeSegment]`. Example usage:

    ```py
    transcription: List[TimeSegment] = [
        LabeledTimeSegment(group="A", label="Knock, knock.", start=0, end=1),
        LabeledTimeSegment(group="B", label="Who's there?", start=2, end=3),
        LabeledTimeSegment(group="A", label="Example.", start=3.5, end=4),
        LabeledTimeSegment(group="B", label="Example who?", start=4.5, end=5.5),
        LabeledTimeSegment(group="A", label="Example illustrating two-person dialogue using `group`.", start=6, end=9),
    ]
    ```
    """

    start: float
    """Start time, in seconds, of this segment."""

    end: float
    """End time, in seconds, of this segment."""

    @staticmethod
    def _data_type() -> _AnnotationType:
        return _AnnotationType.TIME_SEGMENT


@dataclass(frozen=True, config=ValidatorConfig)
class LabeledTimeSegment(TimeSegment):
    """Time segment with accompanying label, e.g. audio transcription."""

    label: str
    """The label associated with this time segment."""


@dataclass(frozen=True, config=ValidatorConfig)
class ScoredTimeSegment(TimeSegment):
    """Time segment with additional float score, representing e.g. model prediction confidence."""

    score: float
    """The score associated with this time segment."""


@dataclass(frozen=True, config=ValidatorConfig)
class ScoredLabeledTimeSegment(TimeSegment):
    """Time segment with accompanying label and score."""

    label: str
    """The label associated with this time segment."""

    score: float
    """The score associated with this time segment."""


@dataclass(frozen=True, config=ValidatorConfig)
class TextSegment(Annotation):
    """
    Represents a segment of text within a specified text field to highlight.
    The `start` index is inclusive and the `end` index is
    exclusive, following the convention of Python string slicing.
    The following example adds highlights to two text fields
    named `text` and `summary`.

    ```py
    text = "Hello, world"
    summary = "A summary of current documentation."

    text_segments: List[TextSegment] = [
        # Highlights "Hello" in "Hello, world"
        TextSegment(text_field="text", start=0, end=5),
        # Highlights "summary of" in "A summary of current documentation."
        TextSegment(text_field="summary", start=2, end=12),
    ]
    ```
    """

    text_field: str
    """Text field column name containing the text segments."""

    start: int
    """Zero-indexed start index (inclusive) of the text segment."""

    end: int
    """Zero-indexed end index (exclusive) of the text segment."""

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"Start index must be non-negative ({self.start} provided)")
        if self.end < 0:
            raise ValueError(f"End index must be non-negative ({self.end} provided)")
        if self.start >= self.end:
            raise ValueError(f"Start index must be less than end index ({self.start} >= {self.end})")

    @staticmethod
    def _data_type() -> _AnnotationType:
        return _AnnotationType.TEXT_SEGMENT


@dataclass(frozen=True, config=ValidatorConfig)
class LabeledTextSegment(TextSegment):
    """Text segment with accompanying label, e.g. Location."""

    label: str
    """The label associated with this text segment."""


@dataclass(frozen=True, config=ValidatorConfig)
class ScoredTextSegment(TextSegment):
    """Text segment with additional float score, representing e.g. model prediction confidence."""

    score: float
    """The score associated with this text segment."""


@dataclass(frozen=True, config=ValidatorConfig)
class ScoredLabeledTextSegment(TextSegment):
    """Text segment with accompanying label and score."""

    label: str
    """The label associated with this text segment."""

    score: float
    """The score associated with this text segment."""


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
    Label,
    ScoredLabel,
    ClassificationLabel,
    ScoredClassificationLabel,
    TimeSegment,
    LabeledTimeSegment,
    ScoredTimeSegment,
    ScoredLabeledTimeSegment,
    TextSegment,
    LabeledTextSegment,
    ScoredTextSegment,
    ScoredLabeledTextSegment,
    Keypoints3D,
    Polyline3D,
]
