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
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from kolena._api.v1.workflow import WorkflowType
from kolena.detection import Inference
from kolena.detection import TestCase
from kolena.detection import TestImage
from kolena.detection import TestSuite
from kolena.detection._datatypes import LoadInferencesDataFrame
from kolena.detection._internal import BaseModel
from kolena.detection.inference import BoundingBox
from kolena.detection.inference import SegmentationMask


class Model(BaseModel):
    """
    The descriptor for your model within the Kolena platform.
    """

    #: Unique name of the model within the platform. If the provided model name has already been registered, that model
    #: and its metadata are loaded upon instantiation.
    name: str

    #: Unstructured metadata associated with the model.
    metadata: Dict[str, Any]

    _TestImageClass = TestImage
    _TestCaseClass = TestCase
    _TestSuiteClass = TestSuite
    _InferenceClass = BoundingBox
    _LoadInferencesDataFrameClass = LoadInferencesDataFrame

    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(name, WorkflowType.DETECTION, metadata)

    def _inferences_from_record(self, record: Any) -> Tuple[TestImage, Optional[List[_InferenceClass]]]:
        test_image = TestImage._from_record(record)
        if record.inferences is None:
            return test_image, None
        inferences = [Inference._from_dict(d) for d in record.inferences]
        bboxes = [inference for inference in inferences if isinstance(inference, (BoundingBox, SegmentationMask))]
        return test_image, bboxes


class InferenceModel(Model):
    """
    Extension of :class:`kolena.detection.Model` with custom ``infer`` method to perform inference.
    """

    #: Function transforming a :class:`kolena.detection.TestImage` into a list of zero or more
    #: :class:`kolena.detection.Inference` objects.
    infer: Callable[[TestImage], Optional[List[Inference]]]

    def __init__(
        self,
        name: str,
        infer: Callable[[TestImage], Optional[List[Inference]]],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        setattr(self, "infer", infer)  # bypass mypy method assignment bug
        super().__init__(name, metadata)
