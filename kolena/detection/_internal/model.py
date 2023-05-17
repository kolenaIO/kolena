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
import dataclasses
import json
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

import pandas as pd
from pydantic import validate_arguments

from kolena._api.v1.core import Model as CoreAPI
from kolena._api.v1.detection import Model as API
from kolena._api.v1.workflow import WorkflowType
from kolena._utils import krequests
from kolena._utils import log
from kolena._utils.batched_load import _BatchedLoader
from kolena._utils.batched_load import DFType
from kolena._utils.consts import BatchSize
from kolena._utils.endpoints import get_model_url
from kolena._utils.frozen import Frozen
from kolena._utils.instrumentation import WithTelemetry
from kolena._utils.serde import from_dict
from kolena._utils.validators import ValidatorConfig
from kolena.detection._internal import BaseTestCase
from kolena.detection._internal import BaseTestImage
from kolena.detection._internal import BaseTestSuite
from kolena.detection._internal.test_image import TestImageType
from kolena.errors import InputValidationError
from kolena.errors import NotFoundError


InferenceType = TypeVar("InferenceType")
SampleInferences = Tuple[TestImageType, Optional[List[InferenceType]]]


class BaseModel(ABC, Frozen, WithTelemetry):
    """
    The descriptor for your model within the Kolena platform.
    """

    #: Unique name of the model within the platform. If the provided model name has already been registered, that model
    #: and its metadata are loaded upon instantiation.
    name: str

    #: Unstructured metadata associated with the model.
    metadata: Dict[str, Any]

    _id: int
    _workflow: WorkflowType

    _TestImageClass: Type[BaseTestImage] = BaseTestImage
    _TestCaseClass: Type[BaseTestCase] = BaseTestCase
    _TestSuiteClass: Type[BaseTestSuite] = BaseTestSuite
    _InferenceClass: Type[InferenceType] = InferenceType
    _LoadInferencesDataFrameClass: Type[DFType] = DFType

    @validate_arguments(config=ValidatorConfig)
    def __init__(self, name: str, workflow: WorkflowType, metadata: Optional[Dict[str, Any]] = None):
        try:
            loaded = self._load_by_name(name)
            if len(loaded.metadata) > 0 and loaded.metadata != metadata:
                log.warn(f"mismatch in model metadata, using loaded metadata (loaded: {loaded.metadata})")
        except NotFoundError:
            loaded = self._create(workflow, name, metadata or {})

        self.name = name
        self.metadata = loaded.metadata
        self._id = loaded.id
        self._workflow = WorkflowType(loaded.workflow)
        self._freeze()

    @classmethod
    @validate_arguments(config=ValidatorConfig)
    def _create(cls, workflow: WorkflowType, name: str, metadata: Dict[str, Any]) -> CoreAPI.EntityData:
        request = CoreAPI.CreateRequest(name=name, metadata=metadata, workflow=workflow.value)
        res = krequests.post(endpoint_path=API.Path.CREATE.value, data=json.dumps(dataclasses.asdict(request)))
        krequests.raise_for_status(res)
        obj = from_dict(data_class=CoreAPI.EntityData, data=res.json())
        log.info(f"created model '{name}' ({get_model_url(obj.id)})")
        return obj

    @classmethod
    @validate_arguments(config=ValidatorConfig)
    def _load_by_name(cls, name: str) -> CoreAPI.EntityData:
        request = CoreAPI.LoadByNameRequest(name=name)
        res = krequests.put(endpoint_path=API.Path.LOAD_BY_NAME.value, data=json.dumps(dataclasses.asdict(request)))
        krequests.raise_for_status(res)
        obj = from_dict(data_class=CoreAPI.EntityData, data=res.json())
        log.info(f"loaded model '{name}' ({get_model_url(obj.id)})")
        return obj

    @validate_arguments(config=ValidatorConfig)
    def load_inferences(
        self,
        test_object: Union[_TestCaseClass, _TestSuiteClass],
    ) -> List[Tuple[_TestImageClass, Optional[List[_InferenceClass]]]]:
        """
        Retrieve the uploaded inferences with identical ground truth labels for each image in a test case or test suite.
        """
        return list(self.iter_inferences(test_object))

    @validate_arguments(config=ValidatorConfig)
    def iter_inferences(
        self,
        test_object: Union[_TestCaseClass, _TestSuiteClass],
    ) -> Iterator[Tuple[_TestImageClass, Optional[List[_InferenceClass]]]]:
        """
        Iterate the uploaded inferences with identical ground truth labels for each image in a test case or test suite.
        """
        for df_batch in self._iter_inference_batch_for_reference(test_object):
            yield from (self._inferences_from_record(record) for record in df_batch.itertuples())

    @validate_arguments(config=ValidatorConfig)
    def _iter_inference_batch_for_reference(
        self,
        test_object: Union[_TestCaseClass, _TestSuiteClass],
        batch_size: int = BatchSize.LOAD_SAMPLES.value,
    ) -> Iterator[_LoadInferencesDataFrameClass]:
        if batch_size <= 0:
            raise InputValidationError(f"invalid batch_size '{batch_size}': expected positive integer")
        test_object_display_type = "test case" if isinstance(test_object, self._TestCaseClass) else "test suite"
        test_object_display_name = f"{test_object_display_type} '{test_object.name}'"
        log.info(f"loading inferences from model '{self.name}' on {test_object_display_name}")
        test_id_key = "test_case_id" if isinstance(test_object, self._TestCaseClass) else "test_suite_id"
        params = dict(model_id=self._id, batch_size=batch_size, **{test_id_key: test_object._id})
        init_request = API.InitLoadInferencesRequest(**params)
        yield from _BatchedLoader.iter_data(
            init_request=init_request,
            endpoint_path=API.Path.INIT_LOAD_INFERENCES.value,
            df_class=self._LoadInferencesDataFrameClass,
        )
        log.info(f"loaded inferences from model '{self.name}' on {test_object_display_name}")

    @validate_arguments(config=ValidatorConfig)
    def load_inferences_by_test_case(
        self,
        test_suite: _TestSuiteClass,
    ) -> Dict[int, List[SampleInferences[_TestImageClass, _InferenceClass]]]:
        """Retrieve the uploaded inferences of a test suite for each image, grouped by test case."""
        batches = list(self._iter_inference_batch_for_test_suite(test_suite))
        df_all = pd.concat(batches)
        df = pd.DataFrame(columns=["test_case_id", "test_sample"])
        df["test_case_id"] = df_all["test_case_id"]
        df["test_sample"] = df_all.apply(lambda record: self._inferences_from_record(record), axis=1)
        df_by_testcase = df.groupby("test_case_id")["test_sample"].agg(list).to_dict()
        return df_by_testcase

    @validate_arguments(config=ValidatorConfig)
    def _iter_inference_batch_for_test_suite(
        self,
        test_suite: _TestSuiteClass,
        batch_size: int = BatchSize.LOAD_SAMPLES.value,
    ) -> Iterator[_LoadInferencesDataFrameClass]:
        if batch_size <= 0:
            raise InputValidationError(f"invalid batch_size '{batch_size}': expected positive integer")
        log.info(f"loading inferences from model '{self.name}' on test suite '{test_suite.name}'")
        params = dict(model_id=self._id, batch_size=batch_size, test_suite_id=test_suite._id)
        init_request = API.InitLoadInferencesByTestCaseRequest(**params)
        yield from _BatchedLoader.iter_data(
            init_request=init_request,
            endpoint_path=API.Path.INIT_LOAD_INFERENCES_BY_TEST_CASE.value,
            df_class=self._LoadInferencesDataFrameClass,
        )
        log.info(f"loaded inferences from model '{self.name}' on test suite '{test_suite.name}'")

    @abstractmethod
    def _inferences_from_record(self, record: Any) -> Tuple[_TestImageClass, Optional[List[_InferenceClass]]]:
        ...
