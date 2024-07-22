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
The output from a [`Model`][kolena.workflow.Model]. In other words, a model is a deterministic transformation from a
[`TestSample`][kolena.workflow.TestSample] to an [`Inference`][kolena.workflow.Inference].

```python
from dataclasses import dataclass
from typing import Optional

from kolena.workflow import Inference
from kolena.workflow.annotation import Keypoints

@dataclass(frozen=True)
class PoseEstimate(Inference):
    skeleton: Optional[Keypoints] = None  # leave empty if nothing is detected
    confidence: Optional[float] = None
```
"""
from typing import Type

from kolena._utils.datatypes import DataObject
from kolena._utils.pydantic_v1.dataclasses import dataclass
from kolena._utils.validators import ValidatorConfig
from kolena.workflow import Composite
from kolena.workflow import TestSample
from kolena.workflow._validators import get_data_object_field_types
from kolena.workflow._validators import safe_issubclass
from kolena.workflow._validators import validate_data_object_type
from kolena.workflow._validators import validate_field
from kolena.workflow.test_sample import _get_composite_fields


@dataclass(frozen=True, config=ValidatorConfig)
class Inference(DataObject):
    """
    The inference produced by a model.

    Typically the structure of this object closely mirrors the structure of the
    [`GroundTruth`][kolena.workflow.GroundTruth] for a workflow, but this is not a requirement.

    During evaluation, the [`TestSample`][kolena.workflow.TestSample] objects, ground truth objects, and these inference
    objects are provided to the [`Evaluator`][kolena.workflow.Evaluator] implementation to compute metrics.

    This object may contain any combination of scalars (e.g. `str`, `float`),
    [`Annotation`][kolena.workflow.annotation.Annotation] objects, or lists of these objects.

    A model processing a [`Composite`][kolena.workflow.Composite] test sample can produce an inference result for each
    of its elements. To associate an inference result to each test sample element, put the attributes and/or annotations
    inside a `DataObject` and use the same attribute name as that used in the [`Composite`][kolena.workflow.Composite]
    test sample.

    Continue with the example given in [`Composite`][kolena.workflow.Composite], where the `FacePairSample` test sample
    type is defined using a pair of images under the `source` and `target` members, we can design a corresponding
    inference type with image-level annotations defined in the `FaceRegion` object:

    ```python
    from dataclasses import dataclass

    from kolena.workflow import DataObject, Inference
    from kolena.workflow.annotation import BoundingBox, Keypoints

    @dataclass(frozen=True)
    class FaceRegion(DataObject):
        bounding_box: BoundingBox
        keypoints: Keypoints

    @dataclass(frozen=True)
    class FacePair(Inference):
        source: FaceRegion
        target: FaceRegion
        similarity: float
    ```

    This way, it is clear which bounding boxes and keypoints are associated to which image in the test sample.
    """


def _validate_inference_type(test_sample_type: Type[TestSample], inference_type: Type[Inference]) -> None:
    if not issubclass(inference_type, Inference):
        raise ValueError(f"Inference must subclass {Inference.__name__}")

    is_composite = issubclass(test_sample_type, Composite)
    composite_fields = _get_composite_fields(test_sample_type) if is_composite else []

    for field_name, field_value in get_data_object_field_types(inference_type).items():
        if field_name in composite_fields and safe_issubclass(field_value, DataObject):
            validate_data_object_type(field_value)
        else:
            validate_field(field_name, field_value)
