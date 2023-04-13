from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from pydantic import validate_arguments

from kolena._api.v1.workflow import WorkflowType
from kolena._utils.validators import ValidatorConfig
from kolena.classification import TestCase
from kolena.classification import TestImage
from kolena.classification import TestSuite
from kolena.detection import Inference
from kolena.detection._datatypes import LoadInferencesDataFrame
from kolena.detection._internal import BaseModel
from kolena.detection.inference import ClassificationLabel


class Model(BaseModel):
    """
    The descriptor for a classification model in the Kolena platform.
    """

    #: Unique name of the model, potentially containing information about the architecture, training dataset,
    #: configuration, framework, commit hash, etc.
    name: str

    #: Unstructured metadata associated with the model.
    metadata: Dict[str, Any]

    _TestImageClass = TestImage
    _TestCaseClass = TestCase
    _TestSuiteClass = TestSuite
    _InferenceClass = Tuple[str, float]
    _LoadInferencesDataFrameClass = LoadInferencesDataFrame

    @validate_arguments(config=ValidatorConfig)
    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(name, WorkflowType.CLASSIFICATION, metadata)

    def _inferences_from_record(self, record: Any) -> Tuple[TestImage, Optional[List[_InferenceClass]]]:
        test_image = TestImage._from_record(record)
        if record.inferences is None:
            return test_image, None
        detection_inferences = [Inference._from_dict(d) for d in record.inferences]
        inferences = [
            (inference.label, inference.confidence)
            for inference in detection_inferences
            if isinstance(inference, ClassificationLabel)
        ]
        return test_image, inferences


class InferenceModel(Model):
    """
    A :class:`kolena.classification.Model` with a special :meth:`kolena.classification.InferenceModel.infer` member
    performing inference on a provided :class:`kolena.classification.TestImage`.
    """

    #: A function transforming an input :class:`kolena.classification.TestImage` to zero or more ``(label, confidence)``
    #: tuples representing model predictions.
    infer: Callable[[TestImage], Optional[List[Tuple[str, float]]]]

    @validate_arguments(config=ValidatorConfig)
    def __init__(
        self,
        name: str,
        infer: Callable[[TestImage], Optional[List[Tuple[str, float]]]],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        setattr(self, "infer", infer)  # bypass mypy method assignment bug
        super().__init__(name, metadata)
