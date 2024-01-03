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
from typing import Callable
from typing import Type

import pytest

from kolena.workflow import GroundTruth
from kolena.workflow import Inference
from kolena.workflow import TestSample
from kolena.workflow.workflow import Workflow


@pytest.fixture
def define_workflow_with_name(
    test_sample_type: Type[TestSample],
    ground_truth_type: Type[GroundTruth],
    inference_type: Type[Inference],
) -> Callable[[str], Workflow]:
    def define_workflow_bound(name: str) -> Workflow:
        return Workflow(
            name=name,
            test_sample_type=test_sample_type,
            ground_truth_type=ground_truth_type,
            inference_type=inference_type,
        )

    return define_workflow_bound


@pytest.mark.parametrize(
    "name",
    [
        # whitespace-only
        "",
        " ",
        "        ",
        "\n",
        "\t\n\t",
        # reserved
        "   fr   ",
        "Detection",
        "ClAsSiFiCaTiOn",
        "KEYPOINTS",
    ],
)
def test__name__invalid(define_workflow_with_name: Callable[[str], Workflow], name: str) -> None:
    with pytest.raises(ValueError):
        define_workflow_with_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "test",
        "test again",
        " another test ",
        "more\ttest\nnames",
        "Something Including (special :: characters)",
        "私のワークフロー (My Workflow)",
        "🤣",
        "🤣🤣🤣",
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
    ],
)
def test__name__valid(define_workflow_with_name: Callable[[str], Workflow], name: str) -> None:
    define_workflow_with_name(name)  # does not throw


@pytest.mark.parametrize(
    "name,sanitized",
    [(" test this ", "test this"), ("\n\ntest\n\n", "test"), ("\tname\nwith\nnewlines\t", "name\nwith\nnewlines")],
)
def test__name__strip(define_workflow_with_name: Callable[[str], Workflow], name: str, sanitized: str) -> None:
    assert define_workflow_with_name(name).name == sanitized
