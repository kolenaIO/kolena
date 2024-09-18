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
import time
from abc import ABCMeta
from collections import defaultdict
from contextlib import contextmanager
from typing import DefaultDict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd

from kolena._api.v1.core import BulkProcessStatus
from kolena._api.v1.core import TestCase as CoreAPI
from kolena._api.v1.event import EventAPI
from kolena._api.v1.generic import TestCase as API
from kolena._utils import krequests
from kolena._utils import log
from kolena._utils.batched_load import _BatchedLoader
from kolena._utils.batched_load import init_upload
from kolena._utils.batched_load import upload_data_frame
from kolena._utils.consts import BatchSize
from kolena._utils.consts import FieldName
from kolena._utils.dataframes.validators import validate_df_schema
from kolena._utils.frozen import Frozen
from kolena._utils.instrumentation import with_event
from kolena._utils.pydantic_v1 import validate_arguments
from kolena._utils.pydantic_v1.dataclasses import dataclass
from kolena._utils.serde import from_dict
from kolena._utils.validators import validate_name
from kolena._utils.validators import ValidatorConfig
from kolena.errors import IncorrectUsageError
from kolena.errors import NotFoundError
from kolena.workflow import GroundTruth
from kolena.workflow import TestSample
from kolena.workflow._datatypes import TestCaseEditorDataFrame
from kolena.workflow._datatypes import TestSampleDataFrame
from kolena.workflow._validators import assert_workflows_match
from kolena.workflow.test_sample import _METADATA_KEY
from kolena.workflow.workflow import Workflow


def _to_editor_data_frame(
    edits: List[Tuple[TestSample, Optional[GroundTruth], bool]],
    test_case_name: Optional[str] = None,
) -> TestCaseEditorDataFrame:
    records = [
        (
            test_case_name,
            test_sample._data_type().value,
            test_sample._to_dict(),
            test_sample._to_metadata_dict(),
            ground_truth._to_dict() if ground_truth is not None else None,
            remove,
        )
        for test_sample, ground_truth, remove, in edits
    ]
    columns = [
        "test_case_name",
        "test_sample_type",
        "test_sample",
        "test_sample_metadata",
        "ground_truth",
        "remove",
    ]
    df = pd.DataFrame(records, columns=columns)
    return TestCaseEditorDataFrame(validate_df_schema(df, TestCaseEditorDataFrame.get_schema(), trusted=True))


class TestCase(Frozen, metaclass=ABCMeta):
    """
    A test case holds a list of [test samples][kolena.workflow.TestSample] paired with
    [ground truths][kolena.workflow.GroundTruth] representing a testing dataset or a slice of a testing dataset.

    Rather than importing this class directly, use the `TestCase` type definition returned from
    [`define_workflow`][kolena.workflow.define_workflow.define_workflow].
    """

    workflow: Workflow
    """
    The workflow of this test case. Automatically populated when constructing via test case type returned from
    [`define_workflow`][kolena.workflow.define_workflow.define_workflow].
    """

    name: str
    """The unique name of this test case. Cannot be changed after creation."""

    version: int
    """The version of this test case. A test case's version is automatically incremented whenever it is edited via
    [`TestCase.edit`][kolena.workflow.TestCase.edit]."""

    description: str
    """Free-form, human-readable description of this test case. Can be edited at any time via
    [`TestCase.edit`][kolena.workflow.TestCase.edit]."""

    _id: int

    def __init_subclass__(cls) -> None:
        if not hasattr(cls, "workflow"):
            raise NotImplementedError(f"{cls.__name__} must implement class attribute 'workflow'")
        super().__init_subclass__()

    @validate_arguments(config=ValidatorConfig)
    def __init__(
        self,
        name: str,
        version: Optional[int] = None,
        description: Optional[str] = None,
        test_samples: Optional[List[Tuple[TestSample, GroundTruth]]] = None,
        reset: bool = False,
    ):
        if type(self) == TestCase:
            raise Exception("<TestCase> must be subclassed.")

        validate_name(name, FieldName.TEST_CASE_NAME)
        self._validate_test_samples(test_samples)

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
            self._populate_from_other(self.create(name, description, test_samples))
        self._freeze()

    @classmethod
    def _validate_test_samples(cls, test_samples: Optional[List[Tuple[TestSample, GroundTruth]]] = None) -> None:
        if not test_samples:
            return

        test_sample_type = cls.workflow.test_sample_type
        ground_truth_type = cls.workflow.ground_truth_type
        if any(
            not (isinstance(sample, test_sample_type) and isinstance(gt, ground_truth_type))
            for sample, gt in test_samples
        ):
            raise TypeError(f"test_samples should be list of tuple of ({test_sample_type}, {ground_truth_type})")

    def _populate_from_other(self, other: "TestCase") -> None:
        with self._unfrozen():
            self._id = other._id
            self.name = other.name
            self.version = other.version
            self.description = other.description
            self.workflow = other.workflow

    @classmethod
    def _create_from_data(cls, data: CoreAPI.EntityData) -> "TestCase":
        assert_workflows_match(cls.workflow.name, data.workflow)
        obj = cls.__new__(cls)
        obj._id = data.id
        obj.name = data.name
        obj.workflow = cls.workflow
        obj.version = data.version
        obj.description = data.description
        obj._freeze()
        return obj

    def _hydrate(self, test_samples: List[Tuple[TestSample, GroundTruth]], description: Optional[str] = None) -> None:
        if len(test_samples) == 0:
            log.warn("no test samples provided, unable to populate test case")
            return
        with self.edit(reset=True) as editor:
            if description is not None:
                editor.description(description)
            for test_sample, ground_truth in test_samples:
                editor.add(test_sample, ground_truth)

    @classmethod
    @with_event(event_name=EventAPI.Event.CREATE_TEST_CASE)
    def create(
        cls,
        name: str,
        description: Optional[str] = None,
        test_samples: Optional[List[Tuple[TestSample, GroundTruth]]] = None,
    ) -> "TestCase":
        """
        Create a new test case with the provided name.

        :param name: The name of the new test case to create.
        :param description: Optional free-form description of the test case to create.
        :param test_samples: Optionally specify a set of test samples and ground truths to populate the test case.
        :return: The newly created test case.
        """
        cls._validate_test_samples(test_samples)
        validate_name(name, FieldName.TEST_CASE_NAME)
        request = CoreAPI.CreateRequest(name=name, description=description or "", workflow=cls.workflow.name)
        res = krequests.post(endpoint_path=API.Path.CREATE.value, data=json.dumps(dataclasses.asdict(request)))
        krequests.raise_for_status(res)
        data = from_dict(data_class=CoreAPI.EntityData, data=res.json())
        obj = cls._create_from_data(data)
        log.info(f"created test case '{name}' (v{obj.version})")
        if test_samples is not None:
            obj._hydrate(test_samples)
        return obj

    @classmethod
    @with_event(event_name=EventAPI.Event.LOAD_TEST_CASE)
    def load(cls, name: str, version: Optional[int] = None) -> "TestCase":
        """
        Load an existing test case with the provided name.

        :param name: The name of the test case to load.
        :param version: Optionally specify a particular version of the test case to load. Defaults to the latest version
            when unset.
        :return: The loaded test case.
        """
        request = CoreAPI.LoadByNameRequest(name=name, version=version)
        res = krequests.put(endpoint_path=API.Path.LOAD.value, data=json.dumps(dataclasses.asdict(request)))
        krequests.raise_for_status(res)
        data = from_dict(data_class=CoreAPI.EntityData, data=res.json())
        log.info(f"loaded test case '{name}' (v{data.version})")
        return cls._create_from_data(data)

    @with_event(event_name=EventAPI.Event.LOAD_TEST_CASE_SAMPLES)
    def load_test_samples(self) -> List[Tuple[TestSample, GroundTruth]]:
        """
        Load all [`TestSample`s][kolena.workflow.TestSample] and [`GroundTruth`s][kolena.workflow.GroundTruth] contained
        in this test case.

        :return: A list of each test sample, paired with its ground truth, in this test case.
        """
        return list(self.iter_test_samples())

    def iter_test_samples(self) -> Iterator[Tuple[TestSample, GroundTruth]]:
        """
        Iterate through all [`TestSample`s][kolena.workflow.TestSample] and
        [`GroundTruth`s][kolena.workflow.GroundTruth] contained in this test case.

        :return: An iterator yielding each test sample, paired with its ground truth, in this test case.
        """
        log.info(f"loading test samples in test case '{self.name}' (v{self.version})")
        test_sample_type = self.workflow.test_sample_type
        ground_truth_type = self.workflow.ground_truth_type
        init_request = CoreAPI.InitLoadContentsRequest(batch_size=BatchSize.LOAD_SAMPLES.value, test_case_id=self._id)
        for df in _BatchedLoader.iter_data(
            init_request=init_request,
            endpoint_path=API.Path.INIT_LOAD_TEST_SAMPLES.value,
            df_class=TestSampleDataFrame,
        ):
            has_metadata = "test_sample_metadata" in df.columns
            for record in df.itertuples():
                metadata_field = record.test_sample_metadata if has_metadata else {}
                test_sample = test_sample_type._from_dict({**record.test_sample, _METADATA_KEY: metadata_field})
                ground_truth = ground_truth_type._from_dict(record.ground_truth)
                yield test_sample, ground_truth
        log.info(f"loaded test samples in test case '{self.name}' (v{self.version})")

    class Editor:
        @dataclass(frozen=True)
        class _Edit:
            test_sample: TestSample
            ground_truth: Optional[GroundTruth] = None
            remove: bool = False

        _edits: List[_Edit]
        _reset: bool
        _description: str
        _initial_description: str

        @validate_arguments(config=ValidatorConfig)
        def __init__(self, description: str, reset: bool) -> None:
            self._edits = []
            self._reset = reset
            self._description = description
            self._initial_description = description

        @validate_arguments(config=ValidatorConfig)
        def description(self, description: str) -> None:
            """Update the description of the test case."""
            self._description = description

        @validate_arguments(config=ValidatorConfig)
        def add(self, test_sample: TestSample, ground_truth: GroundTruth) -> None:
            """
            Add a test sample to the test case. When the test sample already exists in the test case, its ground truth
            is overwritten with the ground truth provided here.

            :param test_sample: The test sample to add.
            :param ground_truth: The ground truth for the test sample.
            """
            self._edits.append(self._Edit(test_sample, ground_truth=ground_truth))

        @validate_arguments(config=ValidatorConfig)
        def remove(self, test_sample: TestSample) -> None:
            """
            Remove a test sample from the test case. Does nothing if the test sample is not in the test case.

            :param test_sample: The test sample to remove.
            """
            self._edits.append(self._Edit(test_sample, remove=True))

        def _edited(self) -> bool:
            return len(self._edits) > 0 or self._reset or self._description != self._initial_description

        def _to_data_frame(self, test_case_name: Optional[str] = None) -> TestCaseEditorDataFrame:
            return _to_editor_data_frame(
                [(edit.test_sample, edit.ground_truth, edit.remove) for edit in self._edits],
                test_case_name,
            )

    @contextmanager
    @with_event(event_name=EventAPI.Event.EDIT_TEST_CASE)
    def edit(self, reset: bool = False) -> Iterator[Editor]:
        """
        Edit this test case in a context:

        ```python
        with test_case.edit() as editor:
            # perform as many editing actions as desired
            editor.add(...)
            editor.remove(...)
        ```

        Changes are committed to the Kolena platform when the context is exited.

        :param reset: Clear all existing test samples in the test case.
        """
        editor = self.Editor(self.description, reset)

        yield editor

        if not editor._edited():
            return

        log.info(f"editing test case '{self.name}' (v{self.version})")
        init_response = init_upload()
        df_serialized = editor._to_data_frame().as_serializable()
        upload_data_frame(df=df_serialized, load_uuid=init_response.uuid)

        request = CoreAPI.CompleteEditRequest(
            test_case_id=self._id,
            current_version=self.version,
            description=editor._description,
            reset=editor._reset,
            uuid=init_response.uuid,
        )
        complete_res = krequests.put(
            endpoint_path=API.Path.COMPLETE_EDIT.value,
            data=json.dumps(dataclasses.asdict(request)),
        )
        krequests.raise_for_status(complete_res)
        test_case_data = from_dict(data_class=CoreAPI.EntityData, data=complete_res.json())
        self._populate_from_other(self._create_from_data(test_case_data))
        log.success(f"edited test case '{self.name}' (v{self.version})")

    @classmethod
    @with_event(event_name=EventAPI.Event.INIT_MANY_TEST_CASES)
    def init_many(
        cls,
        data: List[Tuple[str, List[Tuple[TestSample, GroundTruth]]]],
        reset: bool = False,
    ) -> List["TestCase"]:
        """
        !!! note "Experimental"

            This function is considered **experimental**, so beware that it is subject to changes
            even in patch releases.

        Create, load or edit multiple test cases.

        ```python
        test_cases = TestCase.init_many([
            ("test-case 1", [(test_sample_1, ground_truth_1), ...]),
            ("test-case 2", [(test_sample_2, ground_truth_2), ...])
        ])

        test_suite = TestSuite("my test suite", test_cases=test_cases)
        ```

        Changes are committed to the Kolena platform together. If there is an error, none of the edits would take
        effect.

        :param data: A list of tuples where each tuple is a test case name and a set of test samples and ground truths
                     tuples for the test case.
        :param reset: If a test case of the same name already exists, overwrite with the provided test_samples.
        :return: The test cases.


        """

        if not hasattr(cls, "workflow"):
            raise NotImplementedError(f"{cls.__name__} must implement class attribute 'workflow'")

        if len({name for name, _ in data}) != len(data):
            raise IncorrectUsageError("Multiple edits to the same test case")

        for _, test_samples in data:
            if test_samples:
                cls._validate_test_samples(test_samples)

        cls._validate_test_samples()

        log.info(f"initializing {len(data)} test cases")
        start = time.time()

        df_test_samples = [
            _to_editor_data_frame(
                [(sample, ground_truth, False) for sample, ground_truth in test_samples],
                name,
            ).as_serializable()
            for name, test_samples in data
            if test_samples
        ]
        df_serialized = pd.concat(df_test_samples) if df_test_samples else pd.DataFrame()
        load_uuid: Optional[str] = None
        if len(df_serialized):
            load_uuid = init_upload().uuid
            upload_data_frame(df=df_serialized, load_uuid=load_uuid)

        request = CoreAPI.BulkProcessRequest(
            test_cases=[CoreAPI.SingleProcessRequest(name=name, reset=reset) for name, _ in data],
            workflow=cls.workflow.name,
            uuid=load_uuid,
        )
        response = krequests.post(
            endpoint_path=API.Path.BULK_PROCESS.value,
            data=json.dumps(dataclasses.asdict(request)),
        )
        krequests.raise_for_status(response)
        bulk_response = from_dict(data_class=CoreAPI.BulkProcessResponse, data=response.json())

        test_cases = []
        statuses: DefaultDict[BulkProcessStatus, int] = defaultdict(int)
        for test_case_data in bulk_response.test_cases:
            test_cases.append(cls._create_from_data(test_case_data.data))
            statuses[test_case_data.status] += 1

        end = time.time()
        status_msg = ", ".join([f"{s.value} {statuses[s]}" for s in BulkProcessStatus])
        log.info(f"initialized {len(data)} test cases: {status_msg} in {end - start:0.3f} seconds")

        return test_cases
