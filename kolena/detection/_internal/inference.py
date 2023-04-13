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
from enum import Enum

from kolena._utils.frozen import Frozen
from kolena._utils.serializable import Serializable


class InferenceType(str, Enum):
    CLASSIFICATION_LABEL = "CLASSIFICATION_LABEL"
    BOUNDING_BOX = "BOUNDING_BOX"
    SEGMENTATION_MASK = "SEGMENTATION_MASK"


class Inference(Serializable, Frozen, metaclass=ABCMeta):
    """
    Base class for an inference associated with an image.

    See concrete implementations :class:`kolena.detection.inference.ClassificationLabel`,
    :class:`kolena.detection.inference.BoundingBox`, :class:`kolena.detection.inference.SegmentationMask`, for details.
    """

    @abstractmethod
    def __init__(self) -> None:
        ...
