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

from deprecation import deprecated
from pydantic import validate_arguments

from kolena._api.v1.fr import TestSuite as API
from kolena._utils import krequests
from kolena._utils import log
from kolena._utils.endpoints import get_test_suite_url
from kolena._utils.frozen import Frozen
from kolena._utils.instrumentation import WithTelemetry
from kolena._utils.serde import from_dict
from kolena._utils.validators import ValidatorConfig
from kolena.errors import NotFoundError
from kolena.fr.test_case import TestCase


class TestSuite(ABC, Frozen, WithTelemetry):
    """
    A test suite groups together one or more test cases.
    """

    #: The unique name of this test suite. Cannot be changed after creation.
    name: str

    #: The version of this test suite. A test suite's version is automatically incremented whenever it is edited via
    #: :meth:`TestSuite.edit`.
    version: int

    #: Free-form, human-readable description of this test suite. Can be edited at any time via :meth:`TestSuite.edit`.
    description: str

    #: The baseline :class:`kolena.fr.TestCase` objects belonging to this test suite
    baseline_test_cases: List[TestCase]

    #: The non-baseline :class:`kolena.fr.TestCase` objects belonging to this test suite
    non_baseline_test_cases: List[TestCase]

    #: The count of images attached to the baseline test cases
    baseline_image_count: int

    #: The count of genuine pair attached to the baseline test cases
    baseline_pair_count_genuine: int

    #: The count of imposter pair attached to the baseline test cases
    baseline_pair_count_imposter: int

    #: Deprecated, use :class:`kolena._api.v1.fr.TestSuite.EntityData` instead
    Data = API.EntityData

    _id: int
    _data: API.EntityData

    @validate_arguments(config=ValidatorConfig)
    def __init__(
        self,
        name: str,
        version: Optional[int] = None,
        description: Optional[str] = None,
        baseline_test_cases: Optional[List[TestCase]] = None,
        non_baseline_test_cases: Optional[List[TestCase]] = None,
        reset: bool = False,
    ):
        try:
            self._populate_from_other(self.load(name, version))
            if description is not None and self.description != description and not reset:
                log.warn("test suite already exists, not updating description when reset=False")
            if baseline_test_cases is not None or non_baseline_test_cases is not None:
                if self.version > 0 and not reset:
                    log.warn("test suite already exists, not updating test cases when reset=False")
                else:
                    self._hydrate(baseline_test_cases, non_baseline_test_cases, description)
        except NotFoundError:
            if version is not None:
                log.warn(f"creating new test suite '{name}', ignoring provided version")
            self._populate_from_other(self.create(name, description, baseline_test_cases, non_baseline_test_cases))
        self._freeze()

    @property
    @deprecated(details="use values on :class:`kolena.fr.TestSuite` directly", deprecated_in="0.57.0")
    def data(self) -> API.EntityData:
        return self._data

    @data.setter
    @deprecated(details="use values on :class:`kolena.fr.TestSuite` directly", deprecated_in="0.57.0")
    def data(self, new_data: API.EntityData) -> None:
        self._data = new_data

    @classmethod
    def create(
        cls,
        name: str,
        description: Optional[str] = None,
        baseline_test_cases: Optional[List[TestCase]] = None,
        non_baseline_test_cases: Optional[List[TestCase]] = None,
    ) -> "TestSuite":
        """
        Create a new test suite with the provided name.

        :param name: the name of the new test suite to create.
        :param description: optional free-form description of the test suite to create.
        :param test_cases: optionally specify a set of test cases to populate the test suite.
        :return: the newly created test suite.
        """
        request = API.CreateRequest(name=name, description=description or "")
        res = krequests.post(endpoint_path=API.Path.CREATE.value, data=json.dumps(dataclasses.asdict(request)))
        krequests.raise_for_status(res)
        data = from_dict(data_class=API.EntityData, data=res.json())
        obj = cls._create_from_data(data)
        log.info(f"created test suite '{name}' (v{obj.version}) ({get_test_suite_url(obj._id)})")
        obj._hydrate(baseline_test_cases, non_baseline_test_cases)
        return obj

    @classmethod
    def load(cls, name: str, version: Optional[int] = None) -> "TestSuite":
        """
        Load an existing test suite with the provided name.

        :param name: the name of the test suite to load.
        :param version: optionally specify a particular version of the test suite to load. Defaults to the latest
            version when unset.
        :return: the loaded test suite.
        """
        return cls._load_by_name(name, version)

    @classmethod
    @deprecated(details="use :meth:`load` instead", deprecated_in="0.57.0")
    def load_by_name(cls, name: str, version: Optional[int] = None) -> "TestSuite":
        """
        Retrieve the existing test suite with the provided name.

        :param name: name of the test suite to retrieve.
        :param version: optionally specify the version of the named test suite to retrieve. When absent the latest
            version of the test suite is returned.
        :return: the retrieved test suite.
        :raises NotFoundError: if the test suite with the provided name doesn't exist.
        """
        return cls.load(name, version)

    @classmethod
    def _load_by_name(cls, name: str, version: Optional[int] = None) -> "TestSuite":
        request = API.LoadByNameRequest(name=name, version=version)
        res = krequests.put(endpoint_path=API.Path.LOAD_BY_NAME.value, data=json.dumps(dataclasses.asdict(request)))
        krequests.raise_for_status(res)
        obj = cls._create_from_data(from_dict(data_class=API.EntityData, data=res.json()))
        log.info(f"loaded test suite '{name}' (v{obj.version}) ({get_test_suite_url(obj._id)})")
        return obj

    def _populate_from_other(self, other: "TestSuite") -> None:
        with self._unfrozen():
            self._id = other._id
            self.name = other.name
            self.version = other.version
            self.description = other.description
            self.baseline_test_cases = other.baseline_test_cases
            self.non_baseline_test_cases = other.non_baseline_test_cases
            self.baseline_image_count = other.baseline_image_count
            self.baseline_pair_count_genuine = other.baseline_pair_count_genuine
            self.baseline_pair_count_imposter = other.baseline_pair_count_imposter
            self.data = other._data

    @classmethod
    def _create_from_data(cls, data: API.EntityData) -> "TestSuite":
        obj = cls.__new__(cls)
        obj._id = data.id
        obj.name = data.name
        obj.version = data.version
        obj.description = data.description
        obj.baseline_test_cases = [TestCase._create_from_data(tc) for tc in data.baseline_test_cases]
        obj.non_baseline_test_cases = [TestCase._create_from_data(tc) for tc in data.non_baseline_test_cases]
        obj.baseline_image_count = data.baseline_image_count
        obj.baseline_pair_count_genuine = data.baseline_pair_count_genuine
        obj.baseline_pair_count_imposter = data.baseline_pair_count_imposter
        obj.data = data
        obj._freeze()
        return obj

    def _hydrate(
        self,
        baseline_test_cases: Optional[List[TestCase]] = None,
        non_baseline_test_cases: Optional[List[TestCase]] = None,
        description: Optional[str] = None,
    ) -> None:
        with self.edit(reset=True) as editor:
            if description is not None:
                editor.description(description)
            if baseline_test_cases is not None:
                for test_case in baseline_test_cases:
                    editor.add(test_case, True)
            if non_baseline_test_cases is not None:
                for test_case in non_baseline_test_cases:
                    editor.add(test_case, False)

    class Editor:
        """
        Interface to edit a test suite. Create with :meth:`TestSuite.edit`.
        """

        _edited: bool
        _reset: bool
        _description: str
        _initial_description: str
        _baseline_test_cases: Dict[str, int]
        _non_baseline_test_cases: Dict[str, int]

        @validate_arguments(config=ValidatorConfig)
        def __init__(self, description: str, reset: bool) -> None:
            self._baseline_test_cases: Dict[str, int] = OrderedDict()  # map from name -> id
            self._non_baseline_test_cases: Dict[str, int] = OrderedDict()  # map from name -> id
            self._reset = reset
            self._description = description
            self._initial_description = description
            self._edited = False

        @validate_arguments(config=ValidatorConfig)
        def description(self, description: str) -> None:
            """
            Update the description of the test suite.

            :param description: the new description of the test suite
            """
            self._description = description
            self._edited = True

        @validate_arguments(config=ValidatorConfig)
        def add(self, test_case: TestCase, is_baseline: Optional[bool] = None) -> None:
            """
            Add a test case to this test suite. If a different version of the test case already exists in this test
            suite, it is replaced and its baseline status will be propagated when is_baseline is unset.

            :param test_case: the test case to add to the test suite.
            :param is_baseline: specify that this test case is a part of the "baseline," i.e. if the samples in this
                test case should contribute to the computation of thresholds within this test suite
            """
            name = test_case.name
            # clean up any previous versions and propagates its baseline status
            set_is_baseline: Optional[bool] = None
            if name in self._baseline_test_cases.keys():
                self._baseline_test_cases.pop(name)
                set_is_baseline = is_baseline if is_baseline is not None else True
            if name in self._non_baseline_test_cases.keys():
                self._non_baseline_test_cases.pop(name)
                set_is_baseline = is_baseline if is_baseline is not None else False

            is_baseline_ret = set_is_baseline or is_baseline or False
            self._add(test_case, is_baseline=is_baseline_ret)

        @validate_arguments(config=ValidatorConfig)
        def remove(self, test_case: TestCase) -> None:
            """
            Remove the provided :class:`kolena.fr.TestCase` from the test suite. Any version of this test case in the
            suite will be removed; the version does not need to match exactly.

            :param test_case: the test case to be removed
            """
            name = test_case.name
            if name in self._baseline_test_cases.keys():
                self._baseline_test_cases.pop(name)
            elif name in self._non_baseline_test_cases.keys():
                self._non_baseline_test_cases.pop(name)
            else:
                raise KeyError(f"test case '{name}' not in test suite")
            self._edited = True

        @deprecated(details="use :meth:`add` instead", deprecated_in="0.57.0")
        @validate_arguments(config=ValidatorConfig)
        def merge(self, test_case: TestCase, is_baseline: Optional[bool] = None) -> None:
            """
            Add the :class:`kolena.fr.TestCase` to the suite. If a test case by this name already exists in the suite,
            replace the previous version of that test case with the newly provided version.

            :param test_case: the test case to be merged into the test suite
            :param is_baseline: optionally specify whether or not this test case should be considered as a part of the
                baseline for this test suite. When not specified, the previous value for ``is_baseline`` for this test
                case in this test suite is propagated forward. Defaults to false if the test case does not already exist
                in this suite.
            """
            self.add(test_case, is_baseline)

        @validate_arguments(config=ValidatorConfig)
        def _add(self, test_case: TestCase, is_baseline: bool = False) -> None:
            name = test_case.name
            if is_baseline:
                self._baseline_test_cases[name] = test_case._id
            else:
                self._non_baseline_test_cases[name] = test_case._id
            self._edited = True

    @contextmanager
    def edit(self, reset: bool = False) -> Iterator[Editor]:
        """
        Context-managed way to perform many modification options on a test suite and commit the results when the context
        is exited, resulting in a single version bump.
        """
        editor = TestSuite.Editor(self.description, reset)
        if not reset:
            for baseline_test_case in self.baseline_test_cases:
                editor.add(baseline_test_case, is_baseline=True)
            for non_baseline_test_case in self.non_baseline_test_cases:
                editor.add(non_baseline_test_case, is_baseline=False)
            editor._edited = False

        yield editor

        # no-op contexts have no effect, do not bump version
        if not editor._edited:
            return

        log.info(f"editing test suite '{self.name}' (v{self.version})")
        request = API.EditRequest(
            test_suite_id=self._id,
            current_version=self.version,
            name=self.name,
            description=editor._description,
            baseline_test_case_ids=list(editor._baseline_test_cases.values()),
            non_baseline_test_case_ids=list(editor._non_baseline_test_cases.values()),
        )
        res = krequests.post(endpoint_path=API.Path.EDIT.value, data=json.dumps(dataclasses.asdict(request)))
        krequests.raise_for_status(res)
        test_suite_data = from_dict(data_class=API.EntityData, data=res.json())
        self._populate_from_other(self._create_from_data(test_suite_data))
        log.success(f"edited test suite '{self.name}' (v{self.version}) ({get_test_suite_url(self._id)})")
