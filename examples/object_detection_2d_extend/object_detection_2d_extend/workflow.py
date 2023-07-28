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
from typing import List

from pydantic.dataclasses import dataclass

from kolena._experimental.object_detection import GroundTruth as BaseGroundTruth
from kolena._experimental.object_detection import Inference as BaseInference
from kolena._experimental.object_detection import TestSample as BaseTestSample
from kolena.workflow import define_workflow
from kolena.workflow.annotation import LabeledBoundingBox


@dataclass(frozen=True)
class ExtendedBoundingBox(LabeledBoundingBox):
    occluded: bool


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    bboxes: List[ExtendedBoundingBox]


@dataclass(frozen=True)
class Inference(BaseInference):
    ...


@dataclass(frozen=True)
class TestSample(BaseTestSample):
    ...


_workflow, TestCase, TestSuite, Model = define_workflow(
    "Object Detection Extend",
    TestSample,
    GroundTruth,
    Inference,
)
