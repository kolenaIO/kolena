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
from collections import defaultdict
from contextlib import contextmanager
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import pandas as pd
from pydantic import validate_arguments

from kolena._api.v1 import core
from kolena._api.v1 import generic
from kolena._utils import krequests
from kolena._utils import log
from kolena._utils.batched_load import _BatchedLoader
from kolena._utils.batched_load import init_upload
from kolena._utils.batched_load import upload_data_frame
from kolena._utils.consts import BatchSize
from kolena._utils.consts import FieldName
from kolena._utils.endpoints import get_dataset_url
from kolena._utils.frozen import Frozen
from kolena._utils.instrumentation import telemetry
from kolena._utils.instrumentation import WithTelemetry
from kolena._utils.serde import from_dict
from kolena._utils.validators import validate_name
from kolena._utils.validators import ValidatorConfig
from kolena.errors import IncorrectUsageError
from kolena.errors import NotFoundError
from kolena.workflow import BasicEvaluatorFunction
from kolena.workflow import Evaluator
from kolena.workflow import GroundTruth
from kolena.workflow import Model
from kolena.workflow import TestSample
from kolena.workflow import Workflow
from kolena.workflow._datatypes import DatasetTestSamplesDataFrame
from kolena.workflow._internal import TestRunnable
from kolena.workflow._validators import assert_workflows_match
from kolena.workflow.test_case import _to_editor_data_frame
from kolena.workflow.test_sample import _METADATA_KEY

TEST_CASE_TAG_TYPE = Dict[str, str]
TEST_CASE_TYPE = Tuple[str, List[Tuple[TestSample, GroundTruth]], TEST_CASE_TAG_TYPE]
# func() -> bool or func() -> (bool, GroundTruth)
TEST_CASE_FUNC = Union[
    Callable[[TestSample, GroundTruth], bool],
    Callable[[TestSample, GroundTruth], Tuple[bool, GroundTruth]],
]
DATASET_SAMPLE_TYPE = List[Tuple[TestSample, GroundTruth]]

TestCase = core.Dataset.TestCaseData


class Dataset(Frozen, WithTelemetry, TestRunnable, metaclass=ABCMeta):
    _id: int
    workflow: Workflow
    name: str
    version: int
    description: str
    tags: Set[str]
    test_cases: List[TestCase]

    @telemetry
    def __init_subclass__(cls) -> None:
        if not hasattr(cls, "workflow"):
            raise NotImplementedError(
                f"{cls.__name__} must implement class attribute 'workflow'",
            )
        super().__init_subclass__()

    @validate_arguments(config=ValidatorConfig)
    def __init__(
        self,
        name: str,
        version: Optional[int] = None,
        description: Optional[str] = None,
        test_samples: Optional[DATASET_SAMPLE_TYPE] = None,
        test_cases: Optional[List[TEST_CASE_FUNC]] = None,
        reset: bool = False,
        tags: Optional[Set[str]] = None,
    ):
        validate_name(name, FieldName.DATASET_NAME)
        self._validate_workflow()

        try:
            other = self.load(name, version)
        except NotFoundError:
            other = self.create(
                name,
                description=description,
                tags=tags,
                test_samples=test_samples,
            )

        self._populate_from_other(other)

        # should_update_test_cases = test_cases is not None and test_cases != self.test_cases
        # can_update_test_cases = reset or self.version == 0
        # if should_update_test_cases and not can_update_test_cases:
        #    log.warn(f"reset=False, not updating test cases on dataset '{self.name}' (v{self.version})")
        to_update = [
            # *(["test cases"] if should_update_test_cases and can_update_test_cases else []),
            *(["description"] if description is not None and description != self.description else []),
            *(["tags"] if tags is not None and tags != self.tags else []),
        ]
        if len(to_update) > 0:
            test_samples = test_samples or [] if reset else []
            log.info(
                f"updating {', '.join(to_update)} on dataset '{self.name}' (v{self.version})",
            )
            # updated_test_cases = test_cases or self.test_cases if can_update_test_cases else self.test_cases
            self._hydrate(test_samples, description=description, tags=tags)

        self._freeze()

    @classmethod
    def _validate_workflow(cls) -> None:
        if cls == Dataset:
            raise IncorrectUsageError(
                "<Dataset> must be subclassed. See `kolena.workflow.dataset.define_workflow`",
            )

    def _populate_from_other(self, other: "Dataset") -> None:
        with self._unfrozen():
            self._id = other._id
            self.name = other.name
            self.version = other.version
            self.description = other.description
            self.test_cases = other.test_cases
            self.tags = other.tags

    @classmethod
    def _create_from_data(cls, data: core.Dataset.EntityData) -> "Dataset":
        assert_workflows_match(cls.workflow.name, data.workflow)
        obj = cls.__new__(cls)
        obj._id = data.id
        obj.name = data.name
        obj.version = data.version
        obj.description = data.description
        obj.test_cases = data.test_cases
        obj.tags = set(data.tags)
        obj._freeze()
        return obj

    def _hydrate(
        self,
        test_samples: List[Tuple[TestSample, GroundTruth]],
        description: Optional[str] = None,
        tags: Optional[Set[str]] = None,
    ) -> None:
        with self.edit(reset=True) as editor:
            if description is not None:
                editor.description(description)
            if tags is not None:
                editor.tags = tags
            editor.add_samples(test_samples)

    @classmethod
    def create(
        cls,
        name: str,
        description: Optional[str] = None,
        test_samples: Optional[DATASET_SAMPLE_TYPE] = None,
        test_case_funcs: Optional[List[TEST_CASE_TYPE]] = None,
        tags: Optional[Set[str]] = None,
    ) -> "Dataset":
        """
        Create a new dataset with the provided name.

        :param name: The name of the new dataset to create.
        :param description: Optional free-form description of the dataset to create.
        :param test_cases: Optionally specify a list of test cases to populate the dataset.
        :param tags: Optionally specify a set of tags to attach to the dataset.
        :return: The newly created dataset.
        """
        cls._validate_workflow()
        validate_name(name, FieldName.DATASET_NAME)

        log.info(f"creating dataset '{name}'")

        load_uuid = None
        test_case_funcs = test_case_funcs or []
        test_cases = [
            core.TestCase.SingleProcessRequest(name=tc_name, tags=tc_tags)
            for tc_name, _, tc_tags in test_case_funcs or []
        ]

        if test_samples:
            base = [(ts, gt, False) for ts, gt in test_samples]
            base_df = _to_editor_data_frame(base).as_serializable()
            test_case_df = [
                _to_editor_data_frame([x for x in base if tc_func(x[0], x[1])], tc_name).as_serializable()
                for tc_name, tc_func, _ in test_case_funcs
            ]
            load_uuid = init_upload().uuid
            upload_data_frame(
                df=pd.concat([base_df, *test_case_df]),
                batch_size=BatchSize.UPLOAD_RECORDS.value,
                load_uuid=load_uuid,
            )

        request = core.Dataset.CreateRequest(
            name=name,
            description=description or "",
            workflow=cls.workflow.name,
            tags=list(tags) if tags else [],
            test_cases=test_cases,
            uuid=load_uuid,
        )
        response = krequests.post(
            endpoint_path=generic.Dataset.Path.CREATE.value,
            data=json.dumps(dataclasses.asdict(request)),
        )
        krequests.raise_for_status(response)
        data = from_dict(data_class=core.Dataset.EntityData, data=response.json())

        obj = cls._create_from_data(data)
        log.info(
            f"created dataset '{name}' (v{obj.version}) ({get_dataset_url(obj._id)})",
        )
        return obj

    @classmethod
    def load(cls, name: str, version: Optional[int] = None) -> "Dataset":
        """
        Load an existing dataset with the provided name.

        :param name: The name of the dataset to load.
        :param version: Optionally specify a particular version of the dataset to load. Defaults to the latest
            version when unset.
        :return: The loaded dataset.
        """
        cls._validate_workflow()
        request = core.Dataset.LoadByNameRequest(name=name, version=version)
        res = krequests.put(
            endpoint_path=generic.Dataset.Path.LOAD,
            data=json.dumps(dataclasses.asdict(request)),
        )
        krequests.raise_for_status(res)
        data = from_dict(data_class=core.Dataset.EntityData, data=res.json())
        obj = cls._create_from_data(data)
        log.info(
            f"loaded dataset '{name}' (v{obj.version}) ({get_dataset_url(obj._id)})",
        )
        return obj

    def load_metrics(
        self,
        model: Model,
        evaluator: Union[Evaluator, BasicEvaluatorFunction, None],
    ) -> core.Dataset.LoadMetricsResponse:
        if evaluator is None:
            evaluator_display_name = None
        elif isinstance(evaluator, Evaluator):
            evaluator_display_name = evaluator.display_name()
        else:
            evaluator_display_name = evaluator.__name__

        load_request = core.Dataset.LoadMetricsRequest(
            id=self._id,
            model_id=model._id,
            evaluator=evaluator_display_name,
        )
        response = krequests.put(generic.Dataset.Path.LOAD_METRICS, json=dataclasses.asdict(load_request))
        response.raise_for_status()

        return from_dict(core.Dataset.LoadMetricsResponse, data=response.json())

    class Editor:
        _test_samples: List[Tuple[TestSample, GroundTruth]]
        #        _test_cases: Dict[str, TEST_CASE_TYPE]
        _reset: bool
        _description: str
        _initial_test_case_ids: List[int]
        _initial_description: str
        _initial_tags: Set[str]

        #: The tags associated with this dataset. Modify this list directly to edit this dataset's tags.
        tags: Set[str]

        @validate_arguments(config=ValidatorConfig)
        def __init__(
            self,
            test_cases: List[TestCase],
            description: str,
            tags: Set[str],
            reset: bool,
        ):
            # self._test_cases = {tc[0]: tc for tc in test_cases} if not reset else {}
            self._reset = reset
            self._test_samples: List[Tuple[TestSample, Optional[GroundTruth], bool]] = []
            self._description = description
            self._initial_test_case_ids = [tc.id for tc in test_cases]
            self._initial_description = description
            self._initial_tags = tags
            self.tags = tags

        @validate_arguments(config=ValidatorConfig)
        def description(self, description: str) -> None:
            """Update the description of the dataset."""
            self._description = description

        # @validate_arguments(config=ValidatorConfig)
        # def add_test_cases(self, test_cases: List[TEST_CASE_TYPE]) -> None:
        #     """
        #     Add a list of test cases to this dataset. If a test case already exists in the dataset, it is updated.
        #
        #     :param test_cases: Test cases to add to the dataset.
        #     """
        #     # self._test_cases = [*(tc for tc in self._test_cases if tc.name != test_case.name), test_case]
        #     self._test_cases.update({tc[0]: tc for tc in test_cases})

        @validate_arguments(config=ValidatorConfig)
        def add_test_samples(
            self,
            test_samples: List[Tuple[TestSample, GroundTruth]],
        ) -> None:
            """
            Add a list of test samples to this dataset.

            :param test_samples: Test samples to add to the dataset.
            """
            self._test_samples.extend([(ts, gt, False) for ts, gt in test_samples])

        # @validate_arguments(config=ValidatorConfig)
        # def remove_test_cases(self, test_cases: List[str]) -> None:
        #     """
        #     Remove test cases from this dataset. Does nothing if a test case is not in the dataset.
        #
        #     :param test_cases: Test cases to remove.
        #     """
        #     tc_names = set(self._test_cases.keys())
        #     for tc in test_cases:
        #         if tc in tc_names:
        #             self._test_cases.pop(tc)

        @validate_arguments(config=ValidatorConfig)
        def remove_test_samples(self, test_samples: List[TestSample]) -> None:
            """
            Remove test samples from this dataset. Does nothing if a test sample is not in the dataset.

            :param test_samples: Test samples to remove.
            """
            self._test_samples.extend([(ts, None, True) for ts in test_samples])

        def _edited(self) -> bool:
            return (
                self._description != self._initial_description
                or self._test_samples
                #                 or self._initial_test_case_ids != set(self._test_cases.keys())
                or self._initial_tags != self.tags
            )

    @contextmanager
    def edit(self, reset: bool = False) -> Iterator[Editor]:
        """
        Edit this dataset in a context:

        ```python
        with dataset.edit() as editor:
            # perform as many editing actions as desired
            editor.add(...)
            editor.remove(...)
        ```

        Changes are committed to the Kolena platform when the context is exited.

        :param reset: Clear any and all test cases currently in the dataset.
        """
        editor = self.Editor(self.test_cases, self.description, self.tags, reset)

        yield editor

        if not editor._edited():
            return

        log.info(f"editing dataset '{self.name}' (v{self.version})")
        request = core.Dataset.EditRequest(
            id=self._id,
            current_version=self.version,
            name=self.name,
            description=editor._description,
            # test_cases=list(editor._test_cases.values()),
            tags=list(editor.tags),
        )
        res = krequests.post(
            endpoint_path=generic.Dataset.Path.EDIT,
            data=json.dumps(dataclasses.asdict(request)),
        )
        krequests.raise_for_status(res)
        dataset_data = from_dict(data_class=core.Dataset.EntityData, data=res.json())
        self._populate_from_other(self._create_from_data(dataset_data))
        log.success(
            f"edited dataset '{self.name}' (v{self.version}) ({get_dataset_url(self._id)})",
        )

    def load_test_samples_by_test_case(self) -> List[Tuple[TestCase, List[TestSample]]]:
        """
        Load test samples for all test cases within this dataset.

        :return: A list of [`TestCase`s][kolena.workflow.TestCase], each paired with the list of
            [`TestSample`s][kolena.workflow.TestSample] it contains.
        """
        test_case_id_to_samples: Dict[int, List[TestSample]] = defaultdict(list)
        for df_batch in _BatchedLoader.iter_data(
            init_request=core.Dataset.LoadTestSamplesRequest(
                id=self._id,
                batch_size=BatchSize.LOAD_SAMPLES,
                by_test_case=True,
            ),
            endpoint_path=generic.Dataset.Path.INIT_LOAD_TEST_SAMPLES,
            df_class=DatasetTestSamplesDataFrame,
        ):
            for record in df_batch.itertuples():
                test_sample = self.workflow.test_sample_type._from_dict(
                    record.test_sample,
                )
                test_case_id_to_samples[record.test_case_id].append(test_sample)

        test_case_id_to_test_case = {tc.id: tc for tc in self.test_cases}
        return [
            (test_case_id_to_test_case[tc_id], samples)
            if test_case_id_to_test_case in self.test_cases
            else (
                TestCase(name="_internal", id=tc_id, version=-1),
                samples,
            )
            for tc_id, samples in test_case_id_to_samples.items()
        ]

    def load_test_samples(self) -> DATASET_SAMPLE_TYPE:
        """
        Load test samples for all test cases within this dataset.

        :return: A list of [`TestCase`s][kolena.workflow.TestCase], each paired with the list of
            [`TestSample`s][kolena.workflow.TestSample] it contains.
        """
        test_samples: List[Tuple[TestSample, GroundTruth]] = []
        for df_batch in _BatchedLoader.iter_data(
            init_request=core.Dataset.LoadTestSamplesRequest(
                id=self._id,
                batch_size=BatchSize.LOAD_SAMPLES,
            ),
            endpoint_path=generic.Dataset.Path.INIT_LOAD_TEST_SAMPLES,
            df_class=DatasetTestSamplesDataFrame,
        ):
            for record in df_batch.itertuples():
                test_sample = self.workflow.test_sample_type._from_dict(
                    {**record.test_sample, _METADATA_KEY: record.test_sample_metadata},
                )
                ground_truth = (
                    self.workflow.ground_truth_type._from_dict(record.ground_truth) if record.ground_truth else None
                )
                test_samples.append((test_sample, ground_truth))

        return test_samples

    def get_test_cases(self) -> List[TestCase]:
        response = krequests.put(generic.Dataset.Path.LOAD_TEST_CASES, json=dict(id=self._id))
        krequests.raise_for_status(response)

        return from_dict(core.Dataset.LoadTestCasesResponse, response.json()).test_cases
