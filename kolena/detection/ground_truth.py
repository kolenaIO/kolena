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
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from pydantic import validate_arguments

from kolena._utils.frozen import Frozen
from kolena._utils.geometry import validate_polygon
from kolena._utils.inference_validators import validate_label
from kolena._utils.serializable import Serializable
from kolena._utils.validators import ValidatorConfig
from kolena.detection._internal.ground_truth import GroundTruthType


class GroundTruth(Serializable, Frozen, metaclass=ABCMeta):
    """
    Base class for ground truths associated with an image.

    See concrete implementations :class:`kolena.detection.ground_truth.ClassificationLabel`,
    :class:`kolena.detection.ground_truth.BoundingBox`, :class:`kolena.detection.ground_truth.SegmentationMask`, for
    details.
    """

    #: Label associated with this ground truth.
    label: str

    #: A ground truth marked as `difficult' indicates that the object is considered difficult to recognize,
    #: and should be ignored when evaluating metrics.
    difficult: bool

    @abstractmethod
    def __init__(self) -> None:
        ...

    @classmethod
    def _from_dict(cls, ground_truth: Dict[str, Any]) -> "GroundTruth":
        data_type = ground_truth["data_type"]
        data_object = ground_truth["data_object"]
        if data_type == GroundTruthType.CLASSIFICATION_LABEL.value:
            return ClassificationLabel(label=data_object["label"], difficult=data_object["difficult"])
        if data_type == GroundTruthType.BOUNDING_BOX.value:
            top_left, bottom_right = data_object["points"]
            return BoundingBox(
                label=data_object["label"],
                top_left=top_left,
                bottom_right=bottom_right,
                difficult=data_object["difficult"],
            )
        if data_type == GroundTruthType.SEGMENTATION_MASK.value:
            return SegmentationMask(
                label=data_object["label"],
                points=data_object["points"],
                difficult=data_object["difficult"],
            )
        raise ValueError(f"invalid dictionary provided, unrecognized data type '{data_type}'")


class ClassificationLabel(GroundTruth):
    """
    Ground truth object representing a classification label.
    """

    #: Classification label to associate with the test sample.
    label: str

    #: A classification label marked as `difficult' indicates that the object is considered difficult to recognize,
    #: and should be ignored when evaluating metrics.
    difficult: bool

    @validate_arguments(config=ValidatorConfig)
    def __init__(self, label: str, difficult: bool = False):
        super().__init__()
        validate_label(label)
        self.label = label
        self.difficult = difficult

    def _to_dict(self) -> Dict[str, Any]:
        return dict(
            data_type=GroundTruthType.CLASSIFICATION_LABEL.value,
            data_object=dict(label=self.label, difficult=self.difficult),
        )


class BoundingBox(GroundTruth):
    """
    Ground truth data object representing a bounding box.

    Point coordinates should be in (x, y) format, as absolute pixel values.
    """

    #: Label to associate with this bounding box.
    label: str

    #: Point in (x, y) pixel coordinates representing the top left corner of the bounding box.
    top_left: Tuple[float, float]

    #: Point in (x, y) pixel coordinates representing the bottom right corner of the bounding box.
    bottom_right: Tuple[float, float]

    #: A bounding box marked as `difficult' indicates that the object is considered difficult to recognize,
    #: and should be ignored when evaluating metrics.
    difficult: bool

    @validate_arguments(config=ValidatorConfig)
    def __init__(
        self,
        label: str,
        top_left: Tuple[float, float],
        bottom_right: Tuple[float, float],
        difficult: bool = False,
    ):
        super().__init__()
        validate_label(label)
        self.label = label
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.difficult = difficult

    def _to_dict(self) -> Dict[str, Any]:
        return dict(
            data_type=GroundTruthType.BOUNDING_BOX.value,
            data_object=dict(label=self.label, points=[self.top_left, self.bottom_right], difficult=self.difficult),
        )


class SegmentationMask(GroundTruth):
    """
    Ground truth data object representing a detection mask.

    Point coordinates should be in (x, y) format, as absolute pixel values.

    :raises ValueError: if fewer than three points are provided
    """

    #: Label to associate with this segmentation mask.
    label: str

    #: Polygon corresponding to the vertices of the segmentation mask. Must have at least three distinct elements, and
    #: may not cross itself or touch itself at any point.
    points: List[Tuple[float, float]]

    #: A segmentation mask marked as `difficult' indicates that the object is considered difficult to recognize,
    #: and should be ignored when evaluating metrics.
    difficult: bool

    @validate_arguments(config=ValidatorConfig)
    def __init__(self, label: str, points: List[Tuple[float, float]], difficult: bool = False):
        super().__init__()
        validate_label(label)
        validate_polygon(points)
        self.label = label
        self.points = points
        self.difficult = difficult

    def _to_dict(self) -> Dict[str, Any]:
        return dict(
            data_type=GroundTruthType.SEGMENTATION_MASK.value,
            data_object=dict(label=self.label, points=self.points, difficult=self.difficult),
        )
