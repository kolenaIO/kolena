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
from kolena.workflow import define_workflow_dataset
from kolena.workflow import test
from kolena.workflow.dataset import TestCase
from tests.integration.helper import with_test_prefix
from tests.integration.workflow.conftest import dummy_ground_truth
from tests.integration.workflow.conftest import dummy_inference
from tests.integration.workflow.conftest import dummy_test_sample
from tests.integration.workflow.conftest import DummyEvaluator
from tests.integration.workflow.dummy import DummyGroundTruth
from tests.integration.workflow.dummy import DummyInference
from tests.integration.workflow.dummy import DummyTestSample

DUMMY_WORKFLOW_NAME = "Dataset workflow ⚽"
DUMMY_WORKFLOW, Dataset, Model = define_workflow_dataset(
    name=DUMMY_WORKFLOW_NAME,
    test_sample_type=DummyTestSample,
    ground_truth_type=DummyGroundTruth,
    inference_type=DummyInference,
)


def assert_test_case(src: TestCase, expected: TestCase, check_id: bool = True):
    if check_id:
        assert src.id == expected.id
    assert src.name == expected.name
    assert src.version == expected.version
    assert src.tags == expected.tags


def test__create() -> None:
    name = with_test_prefix(f"{__file__}::test__create")
    dataset = Dataset.create(name)

    assert dataset.test_cases == []


def test__load() -> None:
    name = with_test_prefix(f"{__file__}::test__load")
    dataset = Dataset.create(name, description=f"{name} desc")
    loaded = Dataset.load(name)

    assert loaded == dataset


def test__update_test_samples() -> None:
    ...


def test__update_test_cases() -> None:
    ...


def test__test() -> None:
    name = with_test_prefix(f"{__file__}::test__test")
    n_samples = 10
    test_samples = [dummy_test_sample(i, "dataset-test-run") for i in range(n_samples)]
    ground_truths = [dummy_ground_truth(i) for i in range(n_samples)]
    dataset = Dataset.create(name, test_samples=list(zip(test_samples, ground_truths)))
    model = Model(name, infer=lambda _: dummy_inference())
    evaluator = DummyEvaluator()
    test(model, dataset, evaluator)
