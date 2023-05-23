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
from collections import OrderedDict
from contextlib import contextmanager
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional

import pandas as pd
from deprecation import deprecated
from pydantic import validate_arguments

from kolena._api.v1.fr import TestCase as API
from kolena._utils import krequests
from kolena._utils import log
from kolena._utils.batched_load import _BatchedLoader
from kolena._utils.batched_load import init_upload
from kolena._utils.batched_load import upload_data_frame
from kolena._utils.consts import BatchSize
from kolena._utils.dataframes.validators import validate_df_schema
from kolena._utils.frozen import Frozen
from kolena._utils.instrumentation import WithTelemetry
from kolena._utils.serde import from_dict
from kolena._utils.validators import ValidatorConfig
from kolena.errors import NotFoundError
from kolena.fr.datatypes import TEST_CASE_COLUMNS
from kolena.fr.datatypes import TestCaseDataFrame
from kolena.fr.datatypes import TestCaseDataFrameSchema
from kolena.fr.datatypes import TestCaseRecord


class TestCase(ABC, Frozen, WithTelemetry):
    """
    A group of test samples that can be added to a :class:`kolena.fr.TestSuite`.

    The test case is the base unit of results computation in the Kolena platform. Metrics are computed by test case.
    """

    #: The unique name of this test case. Cannot be changed after creation.
    name: str

    #: The version of this test case. A test case's version is automatically incremented whenever it is edited via
    #: :meth:`TestCase.edit`.
    version: int

    #: Free-form, human-readable description of this test case. Can be edited at any time via :meth:`TestCase.edit`.
    description: str

    #: The count of images attached to this test case
    image_count: int

    #: The count of genuine image pairs attached to this test case
    pair_count_genuine: int

    #: The count of imposter image pairs attached to this test case
    pair_count_imposter: int

    #: Deprecated, use :class:`kolena._api.v1.fr.TestCase.EntityData` instead
    Data = API.EntityData

    _id: int
    _data: API.EntityData

    def __init__(
        self,
        name: str,
        version: Optional[int] = None,
        description: Optional[str] = None,
        test_samples: Optional[List[TestCaseRecord]] = None,
        reset: bool = False,
    ):
        try:
            self._populate_from_other(self.load(name, version))
            if description is not None and self.description != description and not reset:
                log.warn("test case already exists, not updating description when reset=False")
            if test_samples is not None:
                if self.version > 0 and not reset:
                    log.warn("not updating test samples for test case that has already been edited when reset=False")
                else:
                    self._hydrate(test_samples, description)
        except NotFoundError:
            if version is not None:
                log.warn(f"creating new test case '{name}', ignoring provided version")
            self._populate_from_other(self.create(name, description, test_samples))
        self._freeze()

    @property
    @deprecated(details="use values on :class:`kolena.fr.TestCase` directly", deprecated_in="0.57.0")
    def data(self) -> API.EntityData:
        return self._data

    @data.setter
    @deprecated(details="use values on :class:`kolena.fr.TestCase` directly", deprecated_in="0.57.0")
    def data(self, new_data: API.EntityData) -> None:
        self._data = new_data

    @classmethod
    def create(
        cls,
        name: str,
        description: Optional[str] = None,
        test_samples: Optional[List[TestCaseRecord]] = None,
    ) -> "TestCase":
        """
        Create a new test case with the provided name.

        :param name: the name of the new test case to create.
        :param description: optional free-form description of the test case to create.
        :param test_samples: optionally specify a set of test samples to populate the test case.
        :return: the newly created test case.
        """
        request = API.CreateRequest(name=name, description=description or "")
        res = krequests.post(endpoint_path=API.Path.CREATE.value, data=json.dumps(dataclasses.asdict(request)))
        krequests.raise_for_status(res)
        data = from_dict(data_class=API.EntityData, data=res.json())
        obj = cls._create_from_data(data)
        log.info(f"created test case '{name}' (v{obj.version})")
        if test_samples is not None:
            obj._hydrate(test_samples)
        return obj

    @classmethod
    def load(cls, name: str, version: Optional[int] = None) -> "TestCase":
        """
        Load an existing test case with the provided name.

        :param name: the name of the test case to load.
        :param version: optionally specify a particular version of the test case to load. Defaults to the latest version
            when unset.
        :return: the loaded test case.
        """
        return cls._load_by_name(name, version)

    @classmethod
    @deprecated(details="use :meth:`load` instead", deprecated_in="0.57.0")
    def load_by_name(cls, name: str, version: Optional[int] = None) -> "TestCase":
        """
        Load an existing test case with the provided name.

        :param name: the name of the test case to load
        :param version: optionally specify the target version of the test case to load. When absent, the highest version
            of the test case with the provided name is returned
        :return: the loaded test case
        """
        return cls.load(name, version)

    @classmethod
    def _load_by_name(cls, name: str, version: Optional[int] = None) -> "TestCase":
        request = API.LoadByNameRequest(name=name, version=version)
        res = krequests.put(endpoint_path=API.Path.LOAD_BY_NAME.value, data=json.dumps(dataclasses.asdict(request)))
        krequests.raise_for_status(res)
        data = from_dict(data_class=API.EntityData, data=res.json())
        log.info(f"loaded test case '{name}' (v{data.version})")
        return cls._create_from_data(data)

    def load_data(self) -> TestCaseDataFrame:
        """
        Load all pairs data for a test case.

        :return: a DataFrame containing all pairs defined in this test case
        """
        return _BatchedLoader.concat(self.iter_data(), TestCaseDataFrame)

    @classmethod
    def _create_from_data(cls, data: API.EntityData) -> "TestCase":
        obj = cls.__new__(cls)
        obj._id = data.id
        obj.name = data.name
        obj.version = data.version
        obj.description = data.description
        obj.image_count = data.image_count
        obj.pair_count_genuine = data.pair_count_genuine
        obj.pair_count_imposter = data.pair_count_imposter
        obj.data = data
        obj._freeze()
        return obj

    def _hydrate(self, test_samples: List[TestCaseRecord], description: Optional[str] = None) -> None:
        if len(test_samples) == 0:
            log.warn("no test samples provided, unable to populate test case")
            return
        with self.edit(reset=True) as editor:
            if description is not None:
                editor.description(description)
            for locator_a, locator_b, is_same in test_samples:
                editor.add(locator_a, locator_b, is_same)

    def _populate_from_other(self, other: "TestCase") -> None:
        with self._unfrozen():
            self._id = other._id
            self.name = other.name
            self.version = other.version
            self.description = other.description
            self.image_count = other.image_count
            self.pair_count_genuine = other.pair_count_genuine
            self.pair_count_imposter = other.pair_count_imposter
            self.data = other._data

    class Editor:
        _samples: Dict[str, TestCaseRecord]
        _reset: bool
        _description: str
        _initial_description: str
        _initial_samples: Optional[Dict[str, TestCaseRecord]] = None

        def __init__(self, description: str, reset: bool = False) -> None:
            self._reset = reset
            self._description = description
            self._initial_description = description
            self._samples: Dict[str, TestCaseRecord] = OrderedDict()

        @validate_arguments(config=ValidatorConfig)
        def description(self, description: str) -> None:
            """
            Update the description of this test case.

            :param description: the new test case description
            """
            self._description = description

        @validate_arguments(config=ValidatorConfig)
        def add(self, locator_a: str, locator_b: str, is_same: bool) -> None:
            """
            Add the provided image pair to the test case.

            Note that if the image pair with ``locator_a`` and ``locator_b`` is already defined within the platform,
            the value for ``is_same`` must match the value already defined.

            :param locator_a: the left locator for the image pair
            :param locator_b: the right locator for the image pair
            :param is_same: whether or not these images should be considered a true pair or an imposter pair
            :raises ValueError: the image pair already exists in the test case
            """
            key = self._key(locator_a, locator_b)
            val = (locator_a, locator_b, is_same)
            if val == self._samples.get(key, None):
                log.info(f"no op: {val} already in test case")
                return
            self._samples[key] = val

        @validate_arguments(config=ValidatorConfig)
        def remove(self, locator_a: str, locator_b: str) -> None:
            """
            Remove the provided pair from the test case.

            :param locator_a: the left locator for the image pair
            :param locator_b: the right locator for the image pair
            :raises KeyError: if the provided locator pair is not in the test case
            """
            key = self._key(locator_a, locator_b)
            if key not in self._samples.keys():
                raise KeyError(f"pair not in test case: {locator_a}, {locator_b}")
            self._samples.pop(key)

        @staticmethod
        def _key(locator_a: str, locator_b: str) -> str:
            # newline is guaranteed to not be present in the locators
            return f"{locator_a}\n{locator_b}"

        def _edited(self) -> bool:
            return (
                self._reset or self._description != self._initial_description or self._samples != self._initial_samples
            )

    @contextmanager
    def edit(self, reset: bool = False) -> Iterator[Editor]:
        """
        Edit this test case in a context:

        .. code-block:: python

            with test_case.edit() as editor:
                # perform as many editing actions as desired
                editor.add(...)
                editor.remove(...)

        Changes are committed to the Kolena platform when the context is exited.

        :param reset: clear any and all test samples currently in the test case.
        """
        editor = self.Editor(self.description, reset)

        if not reset:
            df_existing = self.load_data()
            # avoid calling the expensive self.load_data() multiple times
            _initial_samples: Dict[str, TestCaseRecord] = OrderedDict()
            for record in df_existing.itertuples():
                locator_a, locator_b, is_same = record.locator_a, record.locator_b, record.is_same
                test_case_record = (locator_a, locator_b, is_same)
                editor.add(*test_case_record)
                _initial_samples[editor._key(locator_a, locator_b)] = test_case_record
            editor._initial_samples = _initial_samples

        yield editor

        # no-op contexts have no effect, do not bump version
        if not editor._edited():
            return

        log.info(f"editing test case '{self.name}' (v{self.version})")
        init_response = init_upload()
        df = pd.DataFrame(editor._samples.values(), columns=TEST_CASE_COLUMNS)
        df_validated = validate_df_schema(df, TestCaseDataFrameSchema)

        upload_data_frame(df=df_validated, batch_size=BatchSize.UPLOAD_RECORDS.value, load_uuid=init_response.uuid)
        request = API.CompleteEditRequest(
            test_case_id=self._id,
            current_version=self.version,
            name=self.name,
            description=editor._description,
            uuid=init_response.uuid,
        )
        complete_res = krequests.post(
            endpoint_path=API.Path.COMPLETE_EDIT.value,
            data=json.dumps(dataclasses.asdict(request)),
        )
        krequests.raise_for_status(complete_res)
        test_case_data = from_dict(data_class=API.EntityData, data=complete_res.json())
        self._populate_from_other(self._create_from_data(test_case_data))
        log.success(f"edited test case '{self.name}' (v{self.version})")

    @validate_arguments
    def iter_data(self, batch_size: int = 10_000_000) -> Iterator[TestCaseDataFrame]:
        """
        Iterator of DataFrames describing all pairs data for a test case.

        :param batch_size: optionally specify maximum number of rows to be returned in a single DataFrame. By default,
            limits row count to 10_000_000.
        """
        log.info(f"loading image pairs in test case '{self.name}' (v{self.version})")
        init_request = API.InitLoadDataRequest(batch_size=batch_size, test_case_id=self._id)
        yield from _BatchedLoader.iter_data(
            init_request=init_request,
            endpoint_path=API.Path.INIT_LOAD_DATA.value,
            df_class=TestCaseDataFrame,
        )
        log.info(f"loaded image pairs in test case '{self.name}' (v{self.version})")
