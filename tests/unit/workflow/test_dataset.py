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
import pytest

from kolena.errors import IncorrectUsageError
from kolena.workflow import Dataset as BaseDataset
from kolena.workflow import define_workflow_dataset
from kolena.workflow import GroundTruth
from kolena.workflow import Image
from kolena.workflow import Inference

DUMMY_WORKFLOW_NAME = "Dataset workflow"
DUMMY_WORKFLOW, Dataset, Model = define_workflow_dataset(
    name=DUMMY_WORKFLOW_NAME,
    test_sample_type=Image,
    ground_truth_type=GroundTruth,
    inference_type=Inference,
)


def test__validate_subclass() -> None:
    with pytest.raises(IncorrectUsageError):
        BaseDataset("my test")

    with pytest.raises(IncorrectUsageError):
        BaseDataset.create("my test")


def test__validate_name() -> None:
    with pytest.raises(ValueError):
        Dataset("")

    with pytest.raises(ValueError):
        Dataset.create("")
