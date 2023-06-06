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
The ground truth associated with a [`TestSample`][kolena.workflow.TestSample]. Typically, a ground truth will represent
the expected output of a model when given a test sample and will be manually annotated by a human.

A [`TestCase`][kolena.workflow.TestCase] holds a list of test samples (model inputs) paired with ground truths
(expected outputs).
"""
from typing import Type

from pydantic.dataclasses import dataclass

from kolena._utils.validators import ValidatorConfig
from kolena.workflow import Composite
from kolena.workflow import TestSample
from kolena.workflow._datatypes import DataObject
from kolena.workflow._validators import get_data_object_field_types
from kolena.workflow._validators import safe_issubclass
from kolena.workflow._validators import validate_data_object_type
from kolena.workflow._validators import validate_field
from kolena.workflow.test_sample import _get_composite_fields


@dataclass(frozen=True, config=ValidatorConfig)
class GroundTruth(DataObject):
    """
    The ground truth against which a model is evaluated.

    A test case contains one or more [`TestSample`][kolena.workflow.TestSample] objects each paired with a ground truth
    object. During evaluation, these test samples, ground truths, and your model's inferences are provided to the
    [`Evaluator`][kolena.workflow.Evaluator] implementation.

    This object may contain any combination of scalars (e.g. `str`, `float`),
    [`Annotation`][kolena.workflow.annotation.Annotation] objects, or lists of these objects.

    For [`Composite`][kolena.workflow.Composite], each object can contain multiple basic test sample elements. To
    associate a set of attributes and/or annotations as the ground truth to a target test sample element, one can use
    [`DataObject`][kolena.workflow.DataObject] and use the same name as in [`Composite`][kolena.workflow.Composite].

    Continue with the example given in [`Composite`][kolena.workflow.Composite], which takes an image pair as a
    test sample, one can design ground truth as:

    ```python
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
    ```

    This way, it is clear which bounding boxes and keypoints are associated to which image in the test sample.
    """


def _validate_ground_truth_type(
    test_sample_type: Type[TestSample],
    ground_truth_type: Type[GroundTruth],
) -> None:
    if not issubclass(ground_truth_type, GroundTruth):
        raise ValueError(f"Ground truth must subclass {GroundTruth.__name__}")

    is_composite = issubclass(test_sample_type, Composite)
    composite_fields = _get_composite_fields(test_sample_type) if is_composite else []

    for field_name, field_value in get_data_object_field_types(ground_truth_type).items():
        if field_name in composite_fields and safe_issubclass(field_value, DataObject):
            validate_data_object_type(field_value)
        else:
            validate_field(field_name, field_value)
