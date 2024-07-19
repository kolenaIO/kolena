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
import dataclasses
import json
from abc import ABCMeta
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TypeVar

from kolena._api.v1.core import Model as CoreAPI
from kolena._api.v1.event import EventAPI
from kolena._api.v1.generic import Model as API
from kolena._utils import krequests
from kolena._utils import log
from kolena._utils.batched_load import _BatchedLoader
from kolena._utils.consts import BatchSize
from kolena._utils.consts import FieldName
from kolena._utils.endpoints import get_model_url
from kolena._utils.frozen import Frozen
from kolena._utils.instrumentation import with_event
from kolena._utils.pydantic_v1 import validate_arguments
from kolena._utils.serde import from_dict
from kolena._utils.validators import validate_name
from kolena._utils.validators import ValidatorConfig
from kolena.errors import NotFoundError
from kolena.workflow import GroundTruth
from kolena.workflow import Inference
from kolena.workflow import TestCase
from kolena.workflow import TestSample as BaseTestSample
from kolena.workflow._datatypes import TestSampleDataFrame
from kolena.workflow._validators import assert_workflows_match
from kolena.workflow.test_sample import _METADATA_KEY
from kolena.workflow.workflow import Workflow

TestSample = TypeVar("TestSample", bound=BaseTestSample)


class Model(Frozen, metaclass=ABCMeta):
    """
    The descriptor of a model tested on Kolena. A model is a deterministic transformation from
    [`TestSample`][kolena.workflow.TestSample] inputs to [`Inference`][kolena.workflow.Inference] outputs.

    Rather than importing this class directly, use the `Model` type definition returned from
    [`define_workflow`][kolena.workflow.define_workflow.define_workflow].
    """

    workflow: Workflow
    """
    The workflow of this model. Automatically populated when constructing via the model type returned from
    [`define_workflow`][kolena.workflow.define_workflow.define_workflow].
    """

    name: str
    """Unique name of the model."""

    metadata: Dict[str, Any]
    """Unstructured metadata associated with the model."""

    tags: Set[str]
    """Tags associated with this model."""

    infer: Optional[Callable[[TestSample], Inference]]
    """
    Function transforming a [`TestSample`][kolena.workflow.TestSample] for a workflow into an
    [`Inference`][kolena.workflow.Inference] object. Required when using [`test`][kolena.workflow.test] or
    [`TestRun.run`][kolena.workflow.TestRun.run].
    """

    _id: int

    def __init_subclass__(cls) -> None:
        if not hasattr(cls, "workflow"):
            raise NotImplementedError(f"{cls.__name__} must implement class attribute 'workflow'")
        super().__init_subclass__()

    @validate_arguments(config=ValidatorConfig)
    def __init__(
        self,
        name: str,
        infer: Optional[Callable[[TestSample], Inference]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
    ):
        if type(self) == Model:
            raise Exception("<Model> must be subclassed.")
        validate_name(name, FieldName.MODEL_NAME)
        try:
            loaded = self.load(name, infer)
            if len(loaded.metadata.keys()) > 0 and loaded.metadata != metadata:
                log.warn(f"mismatch in model metadata, using loaded metadata (loaded: {loaded.metadata})")
            if len(loaded.tags) > 0 and loaded.tags != tags:
                log.warn(f"mismatch in model tags, using loaded tags (loaded: {loaded.tags})")
        except NotFoundError:
            loaded = self.create(name, infer, metadata, tags)

        self._populate_from_other(loaded)

    @classmethod
    @with_event(event_name=EventAPI.Event.CREATE_MODEL)
    def create(
        cls,
        name: str,
        infer: Optional[Callable[[TestSample], Inference]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
    ) -> "Model":
        """
        Create a new model.

        :param name: The unique name of the new model to create.
        :param infer: Optional inference function for this model.
        :param metadata: Optional unstructured metadata to store with this model.
        :param tags: Optional set of tags to associate with this model.
        :return: The newly created model.
        """
        validate_name(name, FieldName.MODEL_NAME)
        metadata = metadata or {}
        request = CoreAPI.CreateRequest(
            name=name,
            metadata=metadata or {},
            workflow=cls.workflow.name,
            tags=list(tags) if tags is not None else None,
        )
        res = krequests.post(endpoint_path=API.Path.CREATE.value, data=json.dumps(dataclasses.asdict(request)))
        krequests.raise_for_status(res)
        obj = cls._from_data(from_dict(data_class=CoreAPI.EntityData, data=res.json()), infer)
        log.info(f"created model '{name}' ({get_model_url(obj._id)})")
        return obj

    @classmethod
    @with_event(event_name=EventAPI.Event.LOAD_MODEL)
    def load(cls, name: str, infer: Optional[Callable[[TestSample], Inference]] = None) -> "Model":
        """
        Load an existing model.

        :param name: The name of the model to load.
        :param infer: Optional inference function for this model.
        """
        request = CoreAPI.LoadByNameRequest(name=name)
        res = krequests.put(endpoint_path=API.Path.LOAD.value, data=json.dumps(dataclasses.asdict(request)))
        krequests.raise_for_status(res)
        obj = cls._from_data(from_dict(data_class=CoreAPI.EntityData, data=res.json()), infer)
        log.info(f"loaded model '{name}' ({get_model_url(obj._id)})")
        return obj

    @classmethod
    @with_event(event_name=EventAPI.Event.LOAD_ALL_MODEL)
    def load_all(
        cls,
        *,
        tags: Optional[Set[str]] = None,
    ) -> List["Model"]:
        """
        Load all models with this workflow.

        :param tags: Optionally specify a set of tags to apply as a filter. The loaded models will include only
            models with tags matching each of these specified tags, i.e.
            `model.tags.intersection(tags) == tags`.
        :return: The models within this workflow, filtered by tags when specified.
        """
        request = CoreAPI.LoadAllRequest(workflow=cls.workflow.name, tags=list(tags) if tags is not None else None)
        res = krequests.put(endpoint_path=API.Path.LOAD_ALL.value, data=json.dumps(dataclasses.asdict(request)))
        krequests.raise_for_status(res)
        data = from_dict(data_class=CoreAPI.LoadAllResponse, data=res.json())
        models = [cls._from_data(model) for model in data.models]
        tags_quoted = {f"'{t}'" for t in tags or {}}
        tags_message = f" with tag{'s' if len(tags) > 1 else ''} {', '.join(tags_quoted)}" if tags else ""
        log.info(f"loaded {len(models)} '{cls.workflow.name}' models{tags_message}")
        return models

    @validate_arguments(config=ValidatorConfig)
    def load_inferences(self, test_case: TestCase) -> List[Tuple[TestSample, GroundTruth, Inference]]:
        """
        Load all inferences stored for this model on the provided test case.

        :param test_case: The test case for which to load inferences.
        :return: The ground truths and inferences for all test samples in the test case.
        """
        return list(self.iter_inferences(test_case))

    @validate_arguments(config=ValidatorConfig)
    def iter_inferences(self, test_case: TestCase) -> Iterator[Tuple[TestSample, GroundTruth, Inference]]:
        """
        Iterate over all inferences stored for this model on the provided test case.

        :param test_case: The test case over which to iterate inferences.
        :return: Iterator exposing the ground truths and inferences for all test samples in the test case.
        """
        log.info(f"loading inferences from model '{self.name}' on test case '{test_case.name}'")
        assert_workflows_match(self.workflow.name, test_case.workflow.name)
        for df_batch in _BatchedLoader.iter_data(
            init_request=API.LoadInferencesRequest(
                model_id=self._id,
                test_case_id=test_case._id,
                batch_size=BatchSize.LOAD_SAMPLES.value,
            ),
            endpoint_path=API.Path.LOAD_INFERENCES.value,
            df_class=TestSampleDataFrame,
        ):
            for record in df_batch.itertuples():
                test_sample = self.workflow.test_sample_type._from_dict(
                    {**record.test_sample, _METADATA_KEY: record.test_sample_metadata},
                )
                ground_truth = self.workflow.ground_truth_type._from_dict(record.ground_truth)
                inference = self.workflow.inference_type._from_dict(record.inference)
                yield test_sample, ground_truth, inference
        log.info(f"loaded inferences from model '{self.name}' on test case '{test_case.name}'")

    def _populate_from_other(self, other: "Model") -> None:
        with self._unfrozen():
            self._id = other._id
            self.name = other.name
            self.metadata = other.metadata
            self.tags = other.tags
            self.workflow = other.workflow
            self.infer = other.infer

    @classmethod
    def _from_data(
        cls,
        data: CoreAPI.EntityData,
        infer: Optional[Callable[[TestSample], Inference]] = None,
    ) -> "Model":
        assert_workflows_match(cls.workflow.name, data.workflow)
        obj = cls.__new__(cls)
        obj._id = data.id
        obj.name = data.name
        obj.metadata = data.metadata
        obj.tags = data.tags
        obj.infer = infer  # type: ignore
        obj._freeze()
        return obj
