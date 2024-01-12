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
from dataclasses import field
from typing import Dict

from pydantic import Extra
from pydantic.dataclasses import dataclass

from kolena.workflow import define_workflow
from kolena.workflow import GroundTruth
from kolena.workflow import Image
from kolena.workflow import Inference
from kolena.workflow.annotation import BoundingBox

DUMMY_WORKFLOW_NAME = "Dummy Workflow 🤖 ci poke"


@dataclass(frozen=True, order=True)
class DummyTestSample(Image):
    value: int
    bbox: BoundingBox
    metadata: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, order=True)
class DummyGroundTruth(GroundTruth):
    label: str
    value: int


@dataclass(frozen=True, order=True)
class DummyInference(Inference):
    score: float


class LocalConfig:
    extra = Extra.allow


@dataclass(frozen=True, order=True, config=LocalConfig)
class GrabbagTestSample(Image):
    value: int


@dataclass(frozen=True, order=True, config=LocalConfig)
class GrabbagGroundTruth(GroundTruth):
    label: str
    value: int


@dataclass(frozen=True, order=True, config=LocalConfig)
class GrabbagInference(Inference):
    ...


DUMMY_WORKFLOW, TestCase, TestSuite, Model = define_workflow(
    name=DUMMY_WORKFLOW_NAME,
    test_sample_type=DummyTestSample,
    ground_truth_type=DummyGroundTruth,
    inference_type=DummyInference,
)
