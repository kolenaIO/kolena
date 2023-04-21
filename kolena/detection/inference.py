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
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from pydantic import validate_arguments

from kolena._utils.geometry import validate_polygon
from kolena._utils.inference_validators import validate_confidence
from kolena._utils.inference_validators import validate_label
from kolena._utils.validators import ValidatorConfig
from kolena.detection._internal import Inference as _Inference
from kolena.detection._internal import InferenceType


class Inference(_Inference):
    @classmethod
    def _from_dict(cls, ground_truth: Dict[str, Any]) -> "Inference":
        data_type = ground_truth["data_type"]
        data_object = ground_truth["data_object"]
        if data_type == InferenceType.CLASSIFICATION_LABEL.value:
            return ClassificationLabel(label=data_object["label"], confidence=data_object["confidence"])
        if data_type == InferenceType.BOUNDING_BOX.value:
            top_left, bottom_right = data_object["points"]
            return BoundingBox(
                label=data_object["label"],
                confidence=data_object["confidence"],
                top_left=top_left,
                bottom_right=bottom_right,
            )
        if data_type == InferenceType.SEGMENTATION_MASK.value:
            return SegmentationMask(
                label=data_object["label"],
                confidence=data_object["confidence"],
                points=data_object["points"],
            )
        raise ValueError(f"invalid dictionary provided, unrecognized data type '{data_type}'")


class ClassificationLabel(Inference):
    """
    Inference representing a classification label.
    """

    #: Predicted classification label.
    label: str

    #: Confidence score associated with this inference.
    confidence: float

    @validate_arguments(config=ValidatorConfig)
    def __init__(self, label: str, confidence: float):
        validate_label(label)
        validate_confidence(confidence)
        self.label = label
        self.confidence = confidence

    def _to_dict(self) -> Dict[str, Any]:
        return dict(
            data_type=InferenceType.CLASSIFICATION_LABEL.value,
            data_object=dict(label=self.label, confidence=self.confidence),
        )


class BoundingBox(Inference):
    """
    Inference representing a bounding box.

    Point coordinates should be in (x, y) format, as absolute pixel values.
    """

    #: Label to associate with this bounding box.
    label: str

    #: Confidence score associated with this inference.
    confidence: float

    #: Point in (x, y) pixel coordinates representing the top left corner of the bounding box.
    top_left: Tuple[float, float]

    #: Point in (x, y) pixel coordinates representing the bottom right corner of the bounding box.
    bottom_right: Tuple[float, float]

    @validate_arguments(config=ValidatorConfig)
    def __init__(self, label: str, confidence: float, top_left: Tuple[float, float], bottom_right: Tuple[float, float]):
        validate_label(label)
        validate_confidence(confidence)
        self.label = label
        self.confidence = confidence
        self.top_left = top_left
        self.bottom_right = bottom_right

    def _to_dict(self) -> Dict[str, Any]:
        return dict(
            data_type=InferenceType.BOUNDING_BOX.value,
            data_object=dict(label=self.label, confidence=self.confidence, points=[self.top_left, self.bottom_right]),
        )


class SegmentationMask(Inference):
    """
    Inference data object representing a segmentation mask.

    Point coordinates should be in (x, y) format, as absolute pixel values.

    :raises ValueError: if fewer than three points are provided
    """

    #: Label to associate with this segmentation mask.
    label: str

    #: Confidence score associated with this inference.
    confidence: float

    #: Polygon corresponding to the vertices of the segmentation mask. Must have at least three distinct elements, and
    #: may not cross itself or touch itself at any point.
    points: List[Tuple[float, float]]

    @validate_arguments(config=ValidatorConfig)
    def __init__(self, label: str, confidence: float, points: List[Tuple[float, float]]):
        validate_label(label)
        validate_confidence(confidence)
        validate_polygon(points)
        self.label = label
        self.confidence = confidence
        self.points = points

    def _to_dict(self) -> Dict[str, Any]:
        return dict(
            data_type=InferenceType.SEGMENTATION_MASK.value,
            data_object=dict(label=self.label, confidence=self.confidence, points=self.points),
        )
