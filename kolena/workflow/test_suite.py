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
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type

from kolena._api.v1.core import TestSuite as CoreAPI
from kolena._api.v1.event import EventAPI
from kolena._api.v1.generic import TestSuite as API
from kolena._utils import krequests
from kolena._utils import log
from kolena._utils.batched_load import _BatchedLoader
from kolena._utils.consts import BatchSize
from kolena._utils.consts import FieldName
from kolena._utils.endpoints import get_test_suite_url
from kolena._utils.frozen import Frozen
from kolena._utils.instrumentation import with_event
from kolena._utils.pydantic_v1 import validate_arguments
from kolena._utils.serde import from_dict
from kolena._utils.validators import validate_name
from kolena._utils.validators import ValidatorConfig
from kolena.errors import IncorrectUsageError
from kolena.errors import NotFoundError
from kolena.workflow import TestSample
from kolena.workflow._datatypes import TestSuiteTestSamplesDataFrame
from kolena.workflow._validators import assert_workflows_match
from kolena.workflow.test_case import TestCase
from kolena.workflow.test_sample import _METADATA_KEY
from kolena.workflow.workflow import Workflow


class TestSuite(Frozen, metaclass=ABCMeta):
    """
    A test suite groups together one or more [test cases][kolena.workflow.TestCase]. Typically a test suite represents a
    benchmark test dataset, with test cases representing different meaningful subsets, or slices, or this benchmark.

    Rather than importing this class directly, use the `TestSuite` type definition returned from
    [`define_workflow`][kolena.workflow.define_workflow.define_workflow].
    """

    workflow: Workflow
    """
    The workflow of this test suite. Automatically populated when constructing via test suite type returned from
    [`define_workflow`][kolena.workflow.define_workflow.define_workflow].
    """

    name: str
    """The unique name of this test suite."""

    version: int
    """
    The version of this test suite. A test suite's version is automatically incremented whenever it is edited via
    [`TestSuite.edit`][kolena.workflow.TestSuite.edit].
    """

    description: str
    """
    Free-form, human-readable description of this test suite. Can be edited at any time via
    [`TestSuite.edit`][kolena.workflow.TestSuite.edit].
    """

    test_cases: List[TestCase]
    """The [`TestCase`][kolena.workflow.TestCase] objects belonging to this test suite."""

    tags: Set[str]
    """The tags associated with this test suite."""

    _id: int
    _test_case_type: Type[TestCase]

    def __init_subclass__(cls) -> None:
        if not hasattr(cls, "workflow"):
            raise NotImplementedError(f"{cls.__name__} must implement class attribute 'workflow'")
        if not hasattr(cls, "_test_case_type"):
            raise NotImplementedError(f"{cls.__name__} must implement class attribute '_test_case_type'")
        super().__init_subclass__()

    @validate_arguments(config=ValidatorConfig)
    def __init__(
        self,
        name: str,
        version: Optional[int] = None,
        description: Optional[str] = None,
        test_cases: Optional[List[TestCase]] = None,
        reset: bool = False,
        tags: Optional[Set[str]] = None,
    ):
        validate_name(name, FieldName.TEST_SUITE_NAME)
        self._validate_workflow()
        self._validate_test_cases(test_cases)

        try:
            other = self.load(name, version)
        except NotFoundError:
            other = self.create(name, description, test_cases, tags)
        self._populate_from_other(other)

        should_update_test_cases = test_cases is not None and test_cases != self.test_cases
        can_update_test_cases = reset or self.version == 0
        if should_update_test_cases and not can_update_test_cases:
            log.warn(f"reset=False, not updating test cases on test suite '{self.name}' (v{self.version})")
        to_update = [
            *(["test cases"] if should_update_test_cases and can_update_test_cases else []),
            *(["description"] if description is not None and description != self.description else []),
            *(["tags"] if tags is not None and tags != self.tags else []),
        ]
        if len(to_update) > 0:
            log.info(f"updating {', '.join(to_update)} on test suite '{self.name}' (v{self.version})")
            updated_test_cases = test_cases or self.test_cases if can_update_test_cases else self.test_cases
            self._hydrate(updated_test_cases, description=description, tags=tags)

        self._freeze()

    @classmethod
    def _validate_test_cases(cls, test_cases: Optional[List[TestCase]] = None) -> None:
        if test_cases:
            if any(cls.workflow != testcase.workflow for testcase in test_cases):
                raise TypeError("test case workflow does not match test suite's")

    @classmethod
    def _validate_workflow(cls) -> None:
        if cls == TestSuite:
            raise IncorrectUsageError("<TestSuite> must be subclassed. See `kolena.workflow.define_workflow`")

    def _populate_from_other(self, other: "TestSuite") -> None:
        with self._unfrozen():
            self._id = other._id
            self.name = other.name
            self.version = other.version
            self.description = other.description
            self.test_cases = other.test_cases
            self.tags = other.tags

    @classmethod
    def _create_from_data(cls, data: CoreAPI.EntityData) -> "TestSuite":
        assert_workflows_match(cls.workflow.name, data.workflow)
        obj = cls.__new__(cls)
        obj._id = data.id
        obj.name = data.name
        obj.version = data.version
        obj.description = data.description
        obj.test_cases = [cls._test_case_type._create_from_data(tc) for tc in data.test_cases]
        obj.tags = set(data.tags)
        obj._freeze()
        return obj

    def _hydrate(
        self,
        test_cases: List[TestCase],
        description: Optional[str] = None,
        tags: Optional[Set[str]] = None,
    ) -> None:
        with self.edit(reset=True) as editor:
            if description is not None:
                editor.description(description)
            for test_case in test_cases:
                editor.add(test_case)
            if tags is not None:
                editor.tags = tags

    @classmethod
    @with_event(event_name=EventAPI.Event.CREATE_TEST_SUITE)
    def create(
        cls,
        name: str,
        description: Optional[str] = None,
        test_cases: Optional[List[TestCase]] = None,
        tags: Optional[Set[str]] = None,
    ) -> "TestSuite":
        """
        Create a new test suite with the provided name.

        :param name: The name of the new test suite to create.
        :param description: Optional free-form description of the test suite to create.
        :param test_cases: Optionally specify a list of test cases to populate the test suite.
        :param tags: Optionally specify a set of tags to attach to the test suite.
        :return: The newly created test suite.
        """
        cls._validate_workflow()
        cls._validate_test_cases(test_cases)
        validate_name(name, FieldName.TEST_SUITE_NAME)
        request = CoreAPI.CreateRequest(
            name=name,
            description=description or "",
            workflow=cls.workflow.name,
            tags=list(tags) if tags is not None else None,
        )
        res = krequests.post(endpoint_path=API.Path.CREATE, data=json.dumps(dataclasses.asdict(request)))
        krequests.raise_for_status(res)
        data = from_dict(data_class=CoreAPI.EntityData, data=res.json())
        obj = cls._create_from_data(data)
        log.info(f"created test suite '{name}' (v{obj.version}) ({get_test_suite_url(obj._id)})")
        if test_cases is not None:
            obj._hydrate(test_cases)
        return obj

    @classmethod
    @with_event(event_name=EventAPI.Event.LOAD_TEST_SUITE)
    def load(cls, name: str, version: Optional[int] = None) -> "TestSuite":
        """
        Load an existing test suite with the provided name.

        :param name: The name of the test suite to load.
        :param version: Optionally specify a particular version of the test suite to load. Defaults to the latest
            version when unset.
        :return: The loaded test suite.
        """
        cls._validate_workflow()
        request = CoreAPI.LoadByNameRequest(name=name, version=version)
        res = krequests.put(endpoint_path=API.Path.LOAD, data=json.dumps(dataclasses.asdict(request)))
        krequests.raise_for_status(res)
        data = from_dict(data_class=CoreAPI.EntityData, data=res.json())
        obj = cls._create_from_data(data)
        log.info(f"loaded test suite '{name}' (v{obj.version}) ({get_test_suite_url(obj._id)})")
        return obj

    @classmethod
    @with_event(event_name=EventAPI.Event.LOAD_ALL_TEST_SUITES)
    def load_all(cls, *, tags: Optional[Set[str]] = None) -> List["TestSuite"]:
        """
        Load the latest version of all test suites with this workflow.

        :param tags: Optionally specify a set of tags to apply as a filter. The loaded test suites will include only
            test suites with tags matching each of these specified tags, i.e.
            `test_suite.tags.intersection(tags) == tags`.
        :return: The latest version of all test suites, with matching tags when specified.
        """
        cls._validate_workflow()
        request = CoreAPI.LoadAllRequest(workflow=cls.workflow.name, tags=list(tags) if tags is not None else None)
        res = krequests.put(endpoint_path=API.Path.LOAD_ALL.value, data=json.dumps(dataclasses.asdict(request)))
        krequests.raise_for_status(res)
        data = from_dict(data_class=CoreAPI.LoadAllResponse, data=res.json())
        objs = [cls._create_from_data(test_suite) for test_suite in data.test_suites]
        tags_quoted = {f"'{t}'" for t in tags or {}}
        tags_message = f" with tag{'s' if len(tags) > 1 else ''} {', '.join(tags_quoted)}" if tags else ""
        log.info(f"loaded {len(objs)} '{cls.workflow.name}' test suites{tags_message}")
        return objs

    class Editor:
        _test_cases: List[TestCase]
        _reset: bool
        _description: str
        _initial_test_case_ids: List[int]
        _initial_description: str
        _initial_tags: Set[str]

        #: The tags associated with this test suite. Modify this list directly to edit this test suite's tags.
        tags: Set[str]

        @validate_arguments(config=ValidatorConfig)
        def __init__(self, test_cases: List[TestCase], description: str, tags: Set[str], reset: bool):
            self._test_cases = test_cases if not reset else []
            self._reset = reset
            self._description = description
            self._initial_test_case_ids = [tc._id for tc in test_cases]
            self._initial_description = description
            self._initial_tags = tags
            self.tags = tags

        @validate_arguments(config=ValidatorConfig)
        def description(self, description: str) -> None:
            """Update the description of the test suite."""
            self._description = description

        @validate_arguments(config=ValidatorConfig)
        def add(self, test_case: TestCase) -> None:
            """
            Add a test case to this test suite. If a different version of the test case already exists in this test
            suite, it is replaced.

            :param test_case: The test case to add to the test suite.
            """
            self._test_cases = [*(tc for tc in self._test_cases if tc.name != test_case.name), test_case]

        @validate_arguments(config=ValidatorConfig)
        def remove(self, test_case: TestCase) -> None:
            """
            Remove a test case from this test suite. Does nothing if the test case is not in the test suite.

            :param test_case: The test case to remove.
            """
            self._test_cases = [tc for tc in self._test_cases if tc.name != test_case.name]

        def _edited(self) -> bool:
            return (
                self._description != self._initial_description
                or self._initial_test_case_ids != [tc._id for tc in self._test_cases]
                or self._initial_tags != self.tags
            )

    @contextmanager
    @with_event(event_name=EventAPI.Event.EDIT_TEST_SUITE)
    def edit(self, reset: bool = False) -> Iterator[Editor]:
        """
        Edit this test suite in a context:

        ```python
        with test_suite.edit() as editor:
            # perform as many editing actions as desired
            editor.add(...)
            editor.remove(...)
        ```

        Changes are committed to the Kolena platform when the context is exited.

        :param reset: Clear all existing test cases in the test suite.
        """
        editor = self.Editor(self.test_cases, self.description, self.tags, reset)

        yield editor

        if not editor._edited():
            return

        log.info(f"editing test suite '{self.name}' (v{self.version})")
        request = CoreAPI.EditRequest(
            test_suite_id=self._id,
            current_version=self.version,
            name=self.name,
            description=editor._description,
            test_case_ids=[tc._id for tc in editor._test_cases],
            tags=list(editor.tags),
        )
        res = krequests.post(endpoint_path=API.Path.EDIT, data=json.dumps(dataclasses.asdict(request)))
        krequests.raise_for_status(res)
        test_suite_data = from_dict(data_class=CoreAPI.EntityData, data=res.json())
        self._populate_from_other(self._create_from_data(test_suite_data))
        log.success(f"edited test suite '{self.name}' (v{self.version}) ({get_test_suite_url(self._id)})")

    @with_event(event_name=EventAPI.Event.LOAD_TEST_SUITE_SAMPLES)
    def load_test_samples(self) -> List[Tuple[TestCase, List[TestSample]]]:
        """
        Load test samples for all test cases within this test suite.

        :return: A list of [`TestCase`s][kolena.workflow.TestCase], each paired with the list of
            [`TestSample`s][kolena.workflow.TestSample] it contains.
        """
        test_case_id_to_samples: Dict[int, List[TestSample]] = defaultdict(list)
        for df_batch in _BatchedLoader.iter_data(
            init_request=API.LoadTestSamplesRequest(
                test_suite_id=self._id,
                batch_size=BatchSize.LOAD_SAMPLES,
            ),
            endpoint_path=API.Path.INIT_LOAD_TEST_SAMPLES,
            df_class=TestSuiteTestSamplesDataFrame,
        ):
            for record in df_batch.itertuples():
                test_sample = self.workflow.test_sample_type._from_dict(
                    {**record.test_sample, _METADATA_KEY: record.test_sample_metadata},
                )
                test_case_id_to_samples[record.test_case_id].append(test_sample)

        test_case_id_to_test_case = {tc._id: tc for tc in self.test_cases}
        return [(test_case_id_to_test_case[tc_id], samples) for tc_id, samples in test_case_id_to_samples.items()]
