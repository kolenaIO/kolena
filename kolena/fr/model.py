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
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Union

import numpy as np
from pydantic.dataclasses import dataclass

from kolena._api.v1.fr import Model as API
from kolena._utils import krequests
from kolena._utils import log
from kolena._utils.batched_load import _BatchedLoader
from kolena._utils.endpoints import get_model_url
from kolena._utils.serde import from_dict
from kolena._utils.uninstantiable import Uninstantiable
from kolena._utils.validators import ValidatorConfig
from kolena.fr import TestCase
from kolena.fr import TestSuite
from kolena.fr.datatypes import LoadedPairResultDataFrame


class Model(Uninstantiable["Model.Data"]):
    """The descriptor for your model within the Kolena platform."""

    @dataclass(frozen=True, config=ValidatorConfig)
    class Data:
        """The data stored for your model in the platform."""

        id: int
        name: str
        metadata: Dict[str, Any]

    @classmethod
    def create(cls, name: str, metadata: Dict[str, Any]) -> "Model":
        """
        Create a new model with the provided name and metadata.

        :param name: unique name of the new model to create
        :param metadata: unstructured metadata to associate with the model
        :return: the created model
        :raises ValueError: if a model by the provided name already exists
        """
        request = API.CreateRequest(name=name, metadata=metadata)
        res = krequests.post(endpoint_path=API.Path.CREATE.value, data=json.dumps(dataclasses.asdict(request)))
        krequests.raise_for_status(res)
        obj = Model.__factory__(from_dict(data_class=Model.Data, data=res.json()))
        log.info(f"created model '{name}' ({get_model_url(obj.data.id)})")
        return obj

    @classmethod
    def load_by_name(cls, name: str) -> "Model":
        """
        Retrieve the existing model with the provided name.

        :param name: name of the model to retrieve
        :return: the retrieved model
        :raises KeyError: if no model with the provided name exists
        """
        request = API.LoadByNameRequest(name=name)
        res = krequests.put(endpoint_path=API.Path.LOAD_BY_NAME.value, data=json.dumps(dataclasses.asdict(request)))
        krequests.raise_for_status(res)
        obj = Model.__factory__(from_dict(data_class=Model.Data, data=res.json()))
        log.info(f"loaded model '{name}' ({get_model_url(obj.data.id)})")
        return obj

    def _get_load_pair_results_request(
        self,
        test_object: Union[TestSuite, TestSuite.Data, TestCase, TestCase.Data],
    ) -> API.LoadPairResultsRequest:
        if isinstance(test_object, TestSuite):
            return API.LoadPairResultsRequest(model_id=self.data.id, test_suite_id=test_object.data.id)
        elif isinstance(test_object, TestSuite.Data):
            return API.LoadPairResultsRequest(model_id=self.data.id, test_suite_id=test_object.id)
        elif isinstance(test_object, TestCase):
            return API.LoadPairResultsRequest(model_id=self.data.id, test_case_id=test_object.data.id)
        elif isinstance(test_object, TestCase.Data):
            return API.LoadPairResultsRequest(model_id=self.data.id, test_case_id=test_object.id)
        else:
            raise ValueError(f"invalid test object type provided: {type(test_object).__name__}")

    def load_pair_results(
        self,
        test_object: Union[TestSuite, TestSuite.Data, TestCase, TestCase.Data],
    ) -> LoadedPairResultDataFrame:
        """
        Load previously stored pair results for this model on the provided test case or test suite. If this model has
        not been run on the provided test object, a zero-length response is returned. Partial results are returned when
        testing on the requested test case or test suite is incomplete.

        The returned DataFrame has the following relevant fields:

        - ``locator_a``: the locator pointing to the left image in the pair
        - ``locator_b``: the locator pointing to the right image in the pair
        - ``is_same``: boolean indicating if the two images depict the same person or a different person
        - ``image_a_fte``: boolean indicating that the left image failed to enroll (FTE)
        - ``image_b_fte``: boolean indicating that the right image failed to enroll (FTE)
        - ``similarity``: float similarity score between the left and right images. ``NaN`` if either image failed to
          enroll. When multiple similarity scores were provided for a given image pair, only the highest similarity
          score is returned

        :param test_object: the :class:`kolena.fr.TestSuite` or :class:`kolena.fr.TestCase` to load pair results from
        :raises ValueError: if an invalid test object was provided
        :raises RemoteError: if the pair results could not be loaded for any reason
        """
        return _BatchedLoader.concat(self.iter_pair_results(test_object), LoadedPairResultDataFrame)

    def iter_pair_results(
        self,
        test_object: Union[TestSuite, TestSuite.Data, TestCase, TestCase.Data],
        batch_size: int = 10_000_000,
    ) -> Iterator[LoadedPairResultDataFrame]:
        """
        Iterator over DataFrames of previously stored pair results for this model on the provided test case or test
        suite, grouped in batches. If this model has not been run on the provided test object, a zero-length response
        is returned. Partial results are returned when testing on the requested test case or test suite is incomplete.

        See :meth:`load_pair_results` for details on the returned DataFrame.

        :param test_object: the :class:`kolena.fr.TestSuite` or :class:`kolena.fr.TestCase` to load pair results from
        :param batch_size: optionally specify maximum number of rows to be returned in a single DataFrame. By default,
            limits row count to 10_000_000.
        :raises ValueError: if an invalid test object was provided
        :raises RemoteError: if the pair results could not be loaded for any reason
        """
        display_type = "test case" if isinstance(test_object, (TestCase, TestCase.Data)) else "test suite"
        display_name = test_object.data.name if isinstance(test_object, (TestCase, TestSuite)) else test_object.name
        test_object_display_name = f"{display_type} '{display_name}'"
        log.info(f"loading pair results from model '{self.data.name}' on {test_object_display_name}")
        base_load_request = dataclasses.asdict(self._get_load_pair_results_request(test_object))
        init_request = API.InitLoadPairResultsRequest(batch_size=batch_size, **base_load_request)
        yield from _BatchedLoader.iter_data(
            init_request=init_request,
            endpoint_path=API.Path.INIT_LOAD_PAIR_RESULTS.value,
            df_class=LoadedPairResultDataFrame,
        )
        log.info(f"loaded pair results from model '{self.data.name}' on {test_object_display_name}")


class InferenceModel(Model):
    """
    A :class:`kolena.fr.Model` capable of running tests via :meth:`kolena.fr.test`.

    Currently supports extracting a single embedding per image. To extract multiple embeddings per image, see
    :meth:`kolena.fr.TestRun`.
    """

    extract: Callable[[str], Optional[np.ndarray]]
    compare: Callable[[np.ndarray, np.ndarray], float]

    @classmethod
    def create(
        cls,
        name: str,
        extract: Callable[[str], Optional[np.ndarray]],
        compare: Callable[[np.ndarray, np.ndarray], float],
        metadata: Dict[str, Any],
    ) -> "InferenceModel":
        base_model = super().create(name, metadata)
        return cls._from_base(base_model, extract, compare)

    @classmethod
    def load_by_name(
        cls,
        name: str,
        extract: Callable[[str], Optional[np.ndarray]],
        compare: Callable[[np.ndarray, np.ndarray], float],
    ) -> "InferenceModel":
        base_model = super().load_by_name(name)
        return cls._from_base(base_model, extract, compare)

    @staticmethod
    def _from_base(
        base_model: Model,
        extract: Callable[[str], Optional[np.ndarray]],
        compare: Callable[[np.ndarray, np.ndarray], float],
    ) -> "InferenceModel":
        inference_model = InferenceModel.__factory__(base_model.data)
        object.__setattr__(inference_model, "__frozen__", False)
        setattr(inference_model, "extract", extract)  # bypass mypy method assignment bug
        setattr(inference_model, "compare", compare)  # see: https://github.com/python/mypy/issues/2427
        inference_model.__frozen__ = True
        return inference_model
