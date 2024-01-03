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
from typing import List

import pytest

from kolena.detection import Model as DetectionModel
from kolena.errors import WorkflowMismatchError
from kolena.workflow import TestRun
from kolena.workflow.define_workflow import define_workflow
from tests.integration.helper import assert_sorted_list_equal
from tests.integration.helper import with_test_prefix
from tests.integration.workflow.conftest import dummy_inference
from tests.integration.workflow.conftest import DummyConfiguration
from tests.integration.workflow.conftest import DummyEvaluator
from tests.integration.workflow.dummy import DUMMY_WORKFLOW
from tests.integration.workflow.dummy import DummyGroundTruth
from tests.integration.workflow.dummy import DummyInference
from tests.integration.workflow.dummy import DummyTestSample
from tests.integration.workflow.dummy import Model
from tests.integration.workflow.dummy import TestSuite

META_DATA = {"a": "b"}
TAGS = {"c", "d"}


def assert_model(model: Model, name: str) -> None:
    assert model.workflow == DUMMY_WORKFLOW
    assert model.name == name
    assert model.metadata == META_DATA
    assert model.tags == TAGS


def test__create() -> None:
    name = with_test_prefix(f"{__file__}::test__create model")
    assert_model(Model.create(name=name, infer=lambda x: None, metadata=META_DATA, tags=TAGS), name)

    with pytest.raises(Exception):
        Model.create(name)


def test__load() -> None:
    name = with_test_prefix(f"{__file__}::test__load model")

    with pytest.raises(Exception):
        Model.load(name)

    Model.create(name, infer=lambda x: None, metadata=META_DATA, tags=TAGS)
    assert_model(Model.load(name, infer=lambda x: None), name)


def test__load_all() -> None:
    name = with_test_prefix(f"{__file__}::test__load_all")
    _, _, _, model = define_workflow(f"{name} workflow 1", DummyTestSample, DummyGroundTruth, DummyInference)
    _, _, _, model_diff = define_workflow(f"{name} workflow 2", DummyTestSample, DummyGroundTruth, DummyInference)
    tag1, tag2, tag1_2 = f"{name} 1", f"{name} 2", f"{name} 1+2"
    model1 = model(name=name + "1", tags={tag1, tag1_2})
    model2 = model(name=name + "2", tags={tag2, tag1_2})
    model3 = model(name=name + "3")
    model_diff(name=name, tags={tag1, tag2, tag1_2})  # Model in different workflow

    test_list = [match for match in zip(model.load_all(), [model1, model2, model3])]
    test_list.extend(zip(model.load_all(tags={tag1_2}), [model1, model2]))
    test_list.extend(zip(model.load_all(tags={tag1, tag1_2}), [model1]))

    for result, expected in test_list:
        assert result.name == expected.name
        assert result.metadata == expected.metadata
        assert result.tags == expected.tags
        assert result.workflow == expected.workflow
    assert model.load_all(tags={"does_not_exist"}) == []


def test__load__mismatching_workflows() -> None:
    name = with_test_prefix(f"{__file__}::test__load__mismatching_workflows")
    DetectionModel(name)
    with pytest.raises(WorkflowMismatchError):
        Model(name)


def test__init() -> None:
    name = with_test_prefix(f"{__file__}::test__init")
    model = Model(name=name, infer=lambda x: None)
    loaded = Model.load(name, infer=lambda x: None)

    assert model.name == loaded.name
    assert model.metadata == loaded.metadata


def test__init_with_optionals() -> None:
    name = with_test_prefix(f"{__file__}::test__init_with_optionals")
    model = Model(name=name, infer=lambda x: None, metadata=META_DATA, tags=TAGS)
    assert_model(model, name)

    with pytest.raises(Exception):
        Model.create(name)

    Model(name=name, infer=lambda x: None, metadata=META_DATA, tags=TAGS)

    assert_model(Model.load(name, infer=lambda x: None), name)

    updated_model = Model(name=name, infer=lambda x: None, metadata={"a": 13}, tags={"e"})
    assert_model(updated_model, name)  # Model metadata and tags don't update


def test__init__validate_name() -> None:
    with pytest.raises(ValueError):
        Model(name=" ", infer=lambda x: None, metadata=META_DATA)


def test__load_inferences__empty(dummy_test_suites: List[TestSuite]) -> None:
    name = with_test_prefix(f"{__file__}::test__load_inferences__empty model")
    model = Model(name)
    assert model.load_inferences(dummy_test_suites[0].test_cases[0]) == []
    assert model.load_inferences(dummy_test_suites[0].test_cases[1]) == []


def test__load_inferences(
    dummy_test_suites: List[TestSuite],
    dummy_test_samples: List[DummyTestSample],
    dummy_ground_truths: List[DummyGroundTruth],
) -> None:
    dummy_inferences = [(dummy_inference(), sample.locator) for sample in dummy_test_samples]
    dummy_inferences_map = {locator: inf for inf, locator in dummy_inferences}
    dummy_inferences_list = [inf for inf, _ in dummy_inferences]

    name = with_test_prefix(f"{__file__}::test__load_inferences model")
    model = Model(name, infer=lambda sample: dummy_inferences_map[sample.locator])

    evaluator = DummyEvaluator(configurations=[DummyConfiguration(value="a")])
    TestRun(model, dummy_test_suites[0], evaluator).run()

    inferences = model.load_inferences(dummy_test_suites[0].test_cases[0])
    assert_sorted_list_equal(inferences, list(zip(dummy_test_samples, dummy_ground_truths, dummy_inferences_list)))
