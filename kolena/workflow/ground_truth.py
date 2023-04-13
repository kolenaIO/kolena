from typing import Type

from pydantic.dataclasses import dataclass

from kolena._utils.validators import ValidatorConfig
from kolena.workflow import Composite
from kolena.workflow import TestSample
from kolena.workflow._datatypes import DataObject
from kolena.workflow._validators import safe_issubclass
from kolena.workflow._validators import validate_data_object_type
from kolena.workflow._validators import validate_field
from kolena.workflow.test_sample import _get_composite_fields


@dataclass(frozen=True, config=ValidatorConfig)
class GroundTruth(DataObject):
    """
    The ground truth against which a model is evaluated.

    A test case contains one or more :class:`kolena.workflow.TestSample` objects each paired with a ground truth object.
    During evaluation, these test samples, ground truths, and your model's inferences are provided to the
    :class:`kolena.workflow.Evaluator` implementation.

    This object may contain any combination of scalars (e.g. ``str``, ``float``),
    :class:`kolena.workflow.annotation.Annotation` objects, or lists of these objects.

    For :class:`kolena.workflow.Composite`, each object can contain multiple basic test sample elements.
    To associate a set of attributes and/or annotations as the ground truth to a target test sample element,
    one can use :class:`kolena.workflow.DataObject` and use the same name as in :class:`kolena.workflow.Composite`.

    Continue with the example given in :class:`kolena.workflow.Composite`, which takes an image pair as a
    test sample, one can design ground truth as:

    .. code-block:: python

        class FacePairSample(kolena.workflow.Composite):
            source: Image
            target: Image


        class FaceRegion(DataObject):
            bounding_box: BoundingBox
            keypoints: Keypoints


        class FacePair(GroundTruth):
            source: FaceRegion
            target: FaceRegion
            is_same_person: bool

    making it clear which bounding boxes and keypoints are associated to which image in the test sample.
    """


def _validate_ground_truth_type(
    test_sample_type: Type[TestSample],
    ground_truth_type: Type[GroundTruth],
) -> None:
    if not issubclass(ground_truth_type, GroundTruth):
        raise ValueError(f"Ground truth must subclass {GroundTruth.__name__}")

    is_composite = issubclass(test_sample_type, Composite)
    composite_fields = _get_composite_fields(test_sample_type) if is_composite else []

    for field_name, field_value in getattr(ground_truth_type, "__annotations__", {}).items():
        if field_name in composite_fields and safe_issubclass(field_value, DataObject):
            validate_data_object_type(field_value)
        else:
            validate_field(field_name, field_value)
