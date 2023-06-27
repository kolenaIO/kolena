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
import uuid
from dataclasses import dataclass
from typing import List
from typing import Tuple

import pytest

from kolena.errors import IncorrectUsageError
from kolena.workflow import define_workflow
from kolena.workflow import GroundTruth
from kolena.workflow import Image
from kolena.workflow import Inference
from kolena.workflow import TestCase
from kolena.workflow import TestSample
from kolena.workflow import Text


@dataclass(frozen=True)
class DummyGroundTruth(GroundTruth):
    ...


@dataclass(frozen=True)
class DummyGroundTruthAlt(GroundTruth):
    ...


@dataclass(frozen=True)
class DummyInference(Inference):
    ...


DummyWorkflow, DummyTestCase, _, _ = define_workflow("Dummy Workflow", Image, DummyGroundTruth, DummyInference)


def test__init__conflict() -> None:
    name = f"{__file__}::test__init__conflict {uuid.uuid4()}"
    with pytest.raises(IncorrectUsageError):
        DummyTestCase.init_many(
            [
                (name, []),
                (name, []),
            ],
        )


@pytest.mark.parametrize(
    "test_samples",
    [
        [[(Image("a"), DummyGroundTruth())], [(Text("a"), DummyGroundTruth())]],
        [[(Image("a"), DummyGroundTruth())], [(Image("b"), DummyGroundTruthAlt())]],
    ],
)
def test__test_sample_validation(test_samples: List[List[Tuple[TestSample, GroundTruth]]]) -> None:
    name = f"{__file__}::test__sample_validation {uuid.uuid4()}"
    with pytest.raises(TypeError):
        DummyTestCase.init_many(
            [(f"{name} {i}", ts) for i, ts in enumerate(test_samples)],
        )


def test__subclassed() -> None:
    with pytest.raises(NotImplementedError):
        TestCase.init_many([])
