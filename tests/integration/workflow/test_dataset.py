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
import random
import string
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import pytest

from kolena.workflow import define_workflow_dataset
from kolena.workflow import test
from kolena.workflow.dataset import DATASET_SAMPLE_TYPE
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


def verify_test_case(src: TestCase, expected: TestCase, check_id: bool = True):
    if check_id:
        assert src.id == expected.id
    assert src.name == expected.name
    assert src.version == expected.version
    assert src.tags == expected.tags


def verify_test_samples(src: DATASET_SAMPLE_TYPE, expected: DATASET_SAMPLE_TYPE) -> None:
    assert sorted(src) == sorted(expected)


def verify_test_case_test_samples(
    src: Dict[int, DATASET_SAMPLE_TYPE],
    expected: Dict[int, DATASET_SAMPLE_TYPE],
) -> None:
    assert len(src) == len(expected) + 1
    base_id = list(set(src.keys()) - set(expected.keys()))[0]
    src = {id: sorted(samples) for id, samples in src.items() if id != base_id}
    expected = {id: sorted(samples) for id, samples in expected.items()}
    assert src == expected


def verify_dataset(
    dataset: Dataset,
    name: str,
    version: int,
    description: Optional[str] = None,
    tags: Optional[Set[str]] = None,
    test_cases: Optional[TestCase] = None,
    test_samples: Optional[Tuple[DummyTestSample, DummyGroundTruth]] = None,
) -> None:
    assert dataset.name == name
    assert dataset.version == version
    assert dataset.workflow == DUMMY_WORKFLOW
    assert dataset.description == (description or "")
    assert dataset.tags == (tags or set())
    assert dataset.test_cases == (test_cases or [])

    verify_test_samples(dataset.load_test_samples(), test_samples or [])


def dummy_samples(
    n_samples: int,
    include_ground_truth: bool = True,
) -> Union[List[DummyTestSample], List[Tuple[DummyTestSample, DummyGroundTruth]]]:
    directory = "".join(random.choices(string.ascii_letters, k=6))
    meta = ["cat", "dog"]
    if include_ground_truth:
        return [
            (dummy_test_sample(i, directory, dict(foo=random.choice(meta))), dummy_ground_truth(i))
            for i in range(n_samples)
        ]
    else:
        return [dummy_test_sample(i, directory, dict(bar=random.choice(meta))) for i in range(n_samples)]


@pytest.mark.parametrize(
    "description,test_samples,tags",
    [
        (None, None, None),
        ("", [], set()),
        ("some", dummy_samples(5), {"world", "cup"}),
    ],
)
def test__create(
    request: pytest.FixtureRequest,
    description: Optional[str],
    test_samples: Optional[List[Tuple[DummyTestSample, DummyGroundTruth]]],
    tags: Optional[Set[str]],
) -> None:
    name = with_test_prefix(f"{__file__}::test__create {request.node.callspec.id}")
    dataset = Dataset.create(name, description=description, tags=tags, test_samples=test_samples)

    assert dataset.test_cases == []
    verify_dataset(dataset, name, 1, description, tags=tags, test_samples=test_samples)


def test__load() -> None:
    name = with_test_prefix(f"{__file__}::test__load")
    dataset = Dataset.create(name, description=f"{name} desc", tags={"esp", "ned"})
    loaded = Dataset.load(name)

    assert loaded == dataset


def test__load_test_samples() -> None:
    name = with_test_prefix(f"{__file__}::test__load_test_samples")
    test_samples = dummy_samples(5)
    dataset = Dataset.create(name, test_samples=test_samples)

    loaded_test_samples = dataset.load_test_samples()

    verify_test_samples(loaded_test_samples, test_samples)

    test_case_test_samples = {tc.id: test_samples for tc in dataset.test_cases}
    loaded_test_case_test_samples = dataset.load_test_samples_by_test_case()
    verify_test_case_test_samples(
        {tc.id: samples for tc, samples in loaded_test_case_test_samples},
        test_case_test_samples,
    )


def test__update_test_samples() -> None:
    name = with_test_prefix(f"{__file__}::test__update_test_samples")
    test_samples = dummy_samples(5)
    dataset = Dataset.create(name, test_samples=test_samples)

    loaded_test_samples = dataset.load_test_samples()

    verify_test_samples(loaded_test_samples, test_samples)

    test_case_test_samples = {tc.id: test_samples for tc in dataset.test_cases}
    loaded_test_case_test_samples = dataset.load_test_samples_by_test_case()
    verify_test_case_test_samples(
        {tc.id: samples for tc, samples in loaded_test_case_test_samples},
        test_case_test_samples,
    )


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

    metrics = dataset.load_metrics(model, evaluator).metrics
    assert len(metrics) == 1
    assert list(metrics.values()) == [dict(value=evaluator.fixed_random_value)]
