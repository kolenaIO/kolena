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
from abc import ABCMeta
from contextlib import contextmanager
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd
from pydantic import validate_arguments
from pydantic.dataclasses import dataclass

from kolena._api.v1.core import TestCase as CoreAPI
from kolena._api.v1.generic import TestCase as API
from kolena._utils import krequests
from kolena._utils import log
from kolena._utils.batched_load import _BatchedLoader
from kolena._utils.batched_load import init_upload
from kolena._utils.batched_load import upload_data_frame
from kolena._utils.consts import BatchSize
from kolena._utils.dataframes.validators import validate_df_schema
from kolena._utils.frozen import Frozen
from kolena._utils.instrumentation import telemetry
from kolena._utils.instrumentation import WithTelemetry
from kolena._utils.serde import from_dict
from kolena._utils.validators import ValidatorConfig
from kolena.errors import NotFoundError
from kolena.workflow import GroundTruth
from kolena.workflow import TestSample
from kolena.workflow._datatypes import TestCaseEditorDataFrame
from kolena.workflow._datatypes import TestSampleDataFrame
from kolena.workflow._validators import assert_workflows_match
from kolena.workflow.workflow import Workflow


class TestCase(Frozen, WithTelemetry, metaclass=ABCMeta):
    """A test case holds a set of images to compute aggregate performance metrics against."""

    #: The :class:`kolena.workflow.Workflow` of this test case.
    workflow: Workflow

    _id: int

    #: The unique name of this test case. Cannot be changed after creation.
    name: str

    #: The version of this test case. A test case's version is automatically incremented whenever it is edited via
    #: :meth:`TestCase.edit`.
    version: int

    #: Free-form, human-readable description of this test case. Can be edited at any time via :meth:`TestCase.edit`.
    description: str

    @telemetry
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
    def create(
        cls,
        name: str,
        description: Optional[str] = None,
        test_samples: Optional[List[Tuple[TestSample, GroundTruth]]] = None,
    ) -> "TestCase":
        """
        Create a new test case with the provided name.

        :param name: the name of the new test case to create.
        :param description: optional free-form description of the test case to create.
        :param test_samples: optionally specify a set of test samples and ground truths to populate the test case.
        :return: the newly created test case.
        """
        cls._validate_test_samples(test_samples)
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
    def load(cls, name: str, version: Optional[int] = None) -> "TestCase":
        """
        Load an existing test case with the provided name.

        :param name: the name of the test case to load.
        :param version: optionally specify a particular version of the test case to load. Defaults to the latest version
            when unset.
        :return: the loaded test case.
        """
        request = CoreAPI.LoadByNameRequest(name=name, version=version)
        res = krequests.put(endpoint_path=API.Path.LOAD.value, data=json.dumps(dataclasses.asdict(request)))
        krequests.raise_for_status(res)
        data = from_dict(data_class=CoreAPI.EntityData, data=res.json())
        log.info(f"loaded test case '{name}' (v{data.version})")
        return cls._create_from_data(data)

    def load_test_samples(self) -> List[Tuple[TestSample, GroundTruth]]:
        """Load all test samples and ground truths in this test case."""
        return list(self.iter_test_samples())

    def iter_test_samples(self) -> Iterator[Tuple[TestSample, GroundTruth]]:
        """Iterate through all test samples and ground truths in this test case."""
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
                test_sample = test_sample_type._from_dict({**record.test_sample, "metadata": metadata_field})
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

            :param test_sample: the test sample to add.
            :param ground_truth: the ground truth for the test sample.
            """
            self._edits.append(self._Edit(test_sample, ground_truth=ground_truth))

        @validate_arguments(config=ValidatorConfig)
        def remove(self, test_sample: TestSample) -> None:
            """
            Remove a test sample from the test case. Does nothing if the test sample is not in the test case.

            :param test_sample: the test sample to remove.
            """
            self._edits.append(self._Edit(test_sample, remove=True))

        def _edited(self) -> bool:
            return len(self._edits) > 0 or self._reset or self._description != self._initial_description

        def _to_data_frame(self) -> TestCaseEditorDataFrame:
            records = [
                (
                    edit.test_sample._data_type().value,
                    edit.test_sample._to_dict(),
                    edit.test_sample._to_metadata_dict(),
                    edit.ground_truth._to_dict() if edit.ground_truth is not None else None,
                    edit.remove,
                )
                for edit in self._edits
            ]
            columns = ["test_sample_type", "test_sample", "test_sample_metadata", "ground_truth", "remove"]
            df = pd.DataFrame(records, columns=columns)
            return TestCaseEditorDataFrame(validate_df_schema(df, TestCaseEditorDataFrame.get_schema(), trusted=True))

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

        yield editor

        if not editor._edited():
            return

        log.info(f"editing test case '{self.name}' (v{self.version})")
        init_response = init_upload()
        df_serialized = editor._to_data_frame().as_serializable()
        upload_data_frame(df=df_serialized, batch_size=BatchSize.UPLOAD_RECORDS.value, load_uuid=init_response.uuid)

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
