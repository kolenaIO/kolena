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
