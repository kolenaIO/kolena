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
from collections import OrderedDict
from contextlib import contextmanager
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Set

from deprecation import deprecated
from pydantic import validate_arguments

import kolena._api.v1.core as CoreAPI
from kolena._api.v1.detection import TestSuite as API
from kolena._api.v1.workflow import WorkflowType
from kolena._utils import krequests
from kolena._utils import log
from kolena._utils.endpoints import get_test_suite_url
from kolena._utils.frozen import Frozen
from kolena._utils.instrumentation import WithTelemetry
from kolena._utils.serde import from_dict
from kolena._utils.validators import ValidatorConfig
from kolena.detection._internal import BaseTestCase
from kolena.errors import NotFoundError
from kolena.workflow._validators import assert_workflows_match


class BaseTestSuite(ABC, Frozen, WithTelemetry):
    """
    A test suite groups together one or more test cases.

    :param name: name of the test suite to create or load
    :param version: optionally specify the version of the test suite to load. When absent, the latest version is loaded.
        Ignored when creating new test suites
    :param description: optionally specify a description for a newly created test suite. For existing test suites, this
        description can be edited via :meth:`TestSuite.edit`
    :param test_cases: optionally specify a list of test cases to populate a new test suite. For existing test suites,
        test cases can be edited via :meth:`TestSuite.edit`
    """

    _id: int
    _workflow: WorkflowType

    @classmethod
    @abstractmethod
    def _test_case_from(cls, response: CoreAPI.TestCase.EntityData) -> BaseTestCase:
        pass

    def __init__(
        self,
        name: str,
        workflow: WorkflowType,
        version: Optional[int] = None,
        description: Optional[str] = None,
        test_cases: Optional[List[BaseTestCase]] = None,
        reset: bool = False,
    ):
        self._validate_test_cases(test_cases)

        try:
            self._populate_from_other(self.load(name, version))
            if description is not None and self.description != description and not reset:
                log.warn("test suite already exists, not updating description when reset=False")
            if test_cases is not None:
                if self.version > 0 and not reset:
                    log.warn("test suite already exists, not updating test cases when reset=False")
                else:
                    self._hydrate(test_cases, description)
        except NotFoundError:
            if version is not None:
                log.warn(f"creating new test suite '{name}', ignoring provided version")
            self._populate_from_other(self._create(workflow, name, description, test_cases))
        self._freeze()

    @classmethod
    def _validate_test_cases(cls, test_cases: Optional[List[BaseTestCase]] = None) -> None:
        if test_cases:
            if any(cls._workflow != testcase._workflow for testcase in test_cases):
                raise TypeError("test case workflow does not match test suite's")

    def _populate_from_other(self, other: "BaseTestSuite") -> None:
        with self._unfrozen():
            self._id = other._id
            self.name = other.name
            self.version = other.version
            self.description = other.description
            self.test_cases = other.test_cases

    @classmethod
    def _create(
        cls,
        workflow: WorkflowType,
        name: str,
        description: Optional[str] = None,
        test_cases: Optional[List[BaseTestCase]] = None,
    ) -> "BaseTestSuite":
        """Create a new test suite with the provided name."""
        request = CoreAPI.TestSuite.CreateRequest(name=name, description=description or "", workflow=workflow.value)
        res = krequests.post(endpoint_path=API.Path.CREATE.value, data=json.dumps(dataclasses.asdict(request)))
        krequests.raise_for_status(res)
        data = from_dict(data_class=CoreAPI.TestSuite.EntityData, data=res.json())
        obj = cls._create_from_data(data)
        log.info(f"created test suite '{name}' (v{obj.version}) ({get_test_suite_url(obj._id)})")
        if test_cases is not None:
            obj._hydrate(test_cases)
        return obj

    @classmethod
    @validate_arguments(config=ValidatorConfig)
    def _load_by_name(cls, name: str, version: Optional[int] = None) -> CoreAPI.TestSuite.EntityData:
        """Retrieve the existing test suite with the provided name."""
        request = CoreAPI.TestSuite.LoadByNameRequest(name=name, version=version)
        data = json.dumps(dataclasses.asdict(request))
        res = krequests.put(endpoint_path=API.Path.LOAD_BY_NAME.value, data=data)
        krequests.raise_for_status(res)
        return from_dict(data_class=CoreAPI.TestSuite.EntityData, data=res.json())

    @validate_arguments(config=ValidatorConfig)
    def _hydrate(self, test_cases: List[BaseTestCase], description: Optional[str] = None) -> None:
        """Convenience method to hydrate an empty test suite with the provided test cases"""
        with self.edit(reset=True) as editor:
            if description is not None:
                editor.description(description)
            if len(test_cases) > 0:
                for test_case in test_cases:
                    editor.add(test_case)

    @classmethod
    def _create_from_data(cls, data: CoreAPI.TestSuite.EntityData) -> "BaseTestSuite":
        assert_workflows_match(cls._workflow, data.workflow)
        obj = cls.__new__(cls)
        obj._id = data.id
        obj.name = data.name
        obj.version = data.version
        obj.description = data.description
        obj.test_cases = [cls._test_case_from(tc) for tc in data.test_cases]
        obj._freeze()
        return obj

    @classmethod
    def create(
        cls,
        name: str,
        description: Optional[str] = None,
        test_cases: Optional[List[BaseTestCase]] = None,
    ) -> "BaseTestSuite":
        """
        Create a new test suite with the provided name.

        :param name: the name of the new test suite to create.
        :param description: optional free-form description of the test suite to create.
        :param test_cases: optionally specify a set of test cases to populate the test suite.
        :return: the newly created test suite.
        """
        return cls._create(cls._workflow, name, description, test_cases)

    @classmethod
    def load(cls, name: str, version: Optional[int] = None) -> "BaseTestSuite":
        """
        Load an existing test suite with the provided name.

        :param name: the name of the test suite to load.
        :param version: optionally specify a particular version of the test suite to load. Defaults to the latest
            version when unset.
        :return: the loaded test suite.
        """
        data = cls._load_by_name(name, version)
        obj = cls._create_from_data(data)
        log.info(f"loaded test suite '{name}' (v{obj.version}) ({get_test_suite_url(obj._id)})")
        return obj

    class Editor:
        """
        Interface to edit a test suite. Create with :meth:`TestSuite.edit`.
        """

        _test_cases: List[BaseTestCase]
        _reset: bool
        _description: str
        _initial_test_case_ids: Set[int]
        _initial_description: str

        _workflow: WorkflowType

        @validate_arguments(config=ValidatorConfig)
        def __init__(self, test_cases: List[BaseTestCase], description: str, reset: bool) -> None:
            self._test_cases: Dict[str, int] = OrderedDict()  # map from name -> id
            self._reset = reset
            self._description = description
            self._initial_test_case_ids = {tc._id for tc in test_cases}
            self._initial_description = description

        @validate_arguments(config=ValidatorConfig)
        def description(self, description: str) -> None:
            """
            Update the description of the test suite.

            :param description: the new description of the test suite
            """
            self._description = description

        @validate_arguments(config=ValidatorConfig)
        def add(self, test_case: BaseTestCase) -> None:
            """
            Add the provided :class:`kolena.detection.TestCase` to the test suite.
            If a different version of the test case already exists in this test suite, it is replaced.

            :param test_case: the test case to add to the test suite
            """
            self._assert_workflows_match(test_case)
            self._test_cases[test_case.name] = test_case._id

        @validate_arguments(config=ValidatorConfig)
        def remove(self, test_case: BaseTestCase) -> None:
            """
            Remove the provided :class:`kolena.detection.TestCase` from the test suite. Any version of this test
            case in this test suite will be removed; the version does not need to match exactly.

            :param test_case: the test case to be removed from the test suite
            """
            name = test_case.name
            if name not in self._test_cases.keys():
                raise KeyError(f"test case '{name}' not in test suite")
            self._test_cases.pop(name)

        @deprecated(details="use :meth:`add` instead", deprecated_in="0.56.0")
        @validate_arguments(config=ValidatorConfig)
        def merge(self, test_case: BaseTestCase) -> None:
            """
            Add the provided :class:`kolena.detection.TestCase` to the test suite, replacing any previous version
            of the test case that may be present in the suite.

            :param test_case: the test case to be merged into the test suite
            """
            self.add(test_case)

        def _assert_workflows_match(self, test_case: BaseTestCase) -> None:
            if test_case._workflow != self._workflow:
                raise ValueError(
                    f"test case workflow '{test_case._workflow}' mismatches test suite workflow '{self._workflow}'",
                )

        def _edited(self) -> bool:
            test_case_ids = set(self._test_cases.values())
            return self._description != self._initial_description or self._initial_test_case_ids != test_case_ids

    @contextmanager
    def edit(self, reset: bool = False) -> Iterator[Editor]:
        """
        Edit this test suite in a context:

        .. code-block:: python

            with test_suite.edit() as editor:
                # perform as many editing actions as desired
                editor.add(...)
                editor.remove(...)

        Changes are committed to the Kolena platform when the context is exited.

        :param reset: clear any and all test cases currently in the test suite.
        """
        editor = BaseTestSuite.Editor(self.test_cases, self.description, reset)
        editor._workflow = self._workflow  # set outside of init such that parameter does not leak into documentation
        if not reset:
            editor.description(self.description)
            for test_case in self.test_cases:
                editor.add(test_case)

        yield editor

        if not editor._edited():
            log.info("no op: nothing edited")
            return

        log.info(f"editing test suite '{self.name}' (v{self.version})")
        request = CoreAPI.TestSuite.EditRequest(
            test_suite_id=self._id,
            current_version=self.version,
            name=self.name,
            description=editor._description,
            test_case_ids=list(editor._test_cases.values()),
        )
        data = json.dumps(dataclasses.asdict(request))
        res = krequests.post(endpoint_path=API.Path.EDIT.value, data=data)
        krequests.raise_for_status(res)
        test_suite_data = from_dict(data_class=CoreAPI.TestSuite.EntityData, data=res.json())
        log.success(f"edited test suite '{self.name}' (v{self.version}) ({get_test_suite_url(test_suite_data.id)})")
        with self._unfrozen():
            self.version = test_suite_data.version
            self.description = test_suite_data.description
            self.test_cases = [self._test_case_from(tc) for tc in test_suite_data.test_cases]
            self._id = test_suite_data.id
