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
from typing import Type

import pytest

from kolena.workflow import GroundTruth
from kolena.workflow import GroundTruth as BaseGroundTruth
from kolena.workflow import Inference
from kolena.workflow import Inference as BaseInference
from kolena.workflow import TestSample
from kolena.workflow import TestSample as BaseTestSample


@pytest.fixture
def test_sample_type() -> Type[TestSample]:
    class ExampleTestSample(BaseTestSample):
        ...

    return ExampleTestSample


@pytest.fixture
def ground_truth_type() -> Type[GroundTruth]:
    class ExampleGroundTruth(BaseGroundTruth):
        ...

    return ExampleGroundTruth


@pytest.fixture
def inference_type() -> Type[Inference]:
    class ExampleInference(BaseInference):
        ...

    return ExampleInference
