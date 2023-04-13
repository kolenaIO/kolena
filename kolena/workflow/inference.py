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
class Inference(DataObject):
    """
    The inference produced by a model.

    Typically the structure of this object closely mirrors the structure of the :class:`kolena.workflow.GroundTruth` for
    a workflow, but this is not a requirement.

    During evaluation, the :class:`kolena.workflow.TestSample` objects, ground truth objects, and these inference
    objects are provided to the :class:`kolena.workflow.Evaluator` implementation to compute metrics.

    This object may contain any combination of scalars (e.g. ``str``, ``float``),
    :class:`kolena.workflow.annotation.Annotation` objects, or lists of these objects.

    A model processing a :class:`kolean.workflow.Composite` object can produce an inference result for each of its
    element. To associate an inference result to each test sample element, one can put the attributes and/or annotations
    inside an :class:`kolena.workflow.DataObject` and use the same name as that in :class:`kolena.workflow.Composite`.

    Continue with the example given in :class:`kolena.workflow.Composite`, which takes an image pair as a
    test sample, one can design inference as:

    .. code-block:: python

        class FacePairSample(kolena.workflow.Composite):
            source: Image
            target: Image


        class FaceRegion(DataObject):
            bounding_box: BoundingBox
            keypoints: Keypoints


        class FacePair(Inference):
            source: FaceRegion
            target: FaceRegion
            similarity: float

    This way, it is clear which bounding boxes and keypoints are associated to which image in the test sample.
    """


def _validate_inference_type(test_sample_type: Type[TestSample], inference_type: Type[Inference]) -> None:
    if not issubclass(inference_type, Inference):
        raise ValueError(f"Inference must subclass {Inference.__name__}")

    is_composite = issubclass(test_sample_type, Composite)
    composite_fields = _get_composite_fields(test_sample_type) if is_composite else []

    for field_name, field_value in getattr(inference_type, "__annotations__", {}).items():
        if field_name in composite_fields and safe_issubclass(field_value, DataObject):
            validate_data_object_type(field_value)
        else:
            validate_field(field_name, field_value)
