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
from typing import cast
from typing import Tuple
from typing import Type

from kolena._utils.pydantic_v1 import validate_arguments
from kolena._utils.validators import ValidatorConfig
from kolena.workflow import GroundTruth
from kolena.workflow import Inference
from kolena.workflow import Model
from kolena.workflow import TestCase
from kolena.workflow import TestSample
from kolena.workflow import TestSuite
from kolena.workflow import Workflow


@validate_arguments(config=ValidatorConfig)
def define_workflow(
    name: str,
    test_sample_type: Type[TestSample],
    ground_truth_type: Type[GroundTruth],
    inference_type: Type[Inference],
) -> Tuple[Workflow, Type[TestCase], Type[TestSuite], Type[Model]]:
    """
    Define a new workflow, specifying its test sample, ground truth, and inference types.

    ```python
    from kolena.workflow import define_workflow

    from my_code import MyTestSample, MyGroundTruth, MyInference

    _, TestCase, TestSuite, Model = define_workflow(
        "My Workflow",
        MyTestSample,   # extends e.g. kolena.workflow.Image (or uses directly)
        MyGroundTruth,  # extends kolena.workflow.GroundTruth
        MyInference,    # extends kolena.workflow.Inference
    )
    ```

    `define_workflow` is provided as a convenience method to create the [`TestCase`][kolena.workflow.TestCase],
    [`TestSuite`][kolena.workflow.TestSuite], and [`Model`][kolena.workflow.Model] objects
    for a new workflow. These objects can also be defined manually by subclassing them and binding the `workflow`
    class variable:

    ```python
    from kolena.workflow import TestCase

    from my_code import my_workflow

    class MyTestCase(TestCase):
        workflow = my_workflow
    ```

    :param name: The name of the workflow.
    :param test_sample_type: The type of the [`TestSample`][kolena.workflow.TestSample] for this workflow.
    :param ground_truth_type: The type of the [`GroundTruth`][kolena.workflow.GroundTruth] for this workflow.
    :param inference_type: The type of the [`Inference`][kolena.workflow.Inference] for this workflow.
    :return: The `Workflow` object for this workflow along with the [`TestCase`][kolena.workflow.TestCase],
        [`TestSuite`][kolena.workflow.TestSuite], and [`Model`][kolena.workflow.Model] objects to use when creating and
        running tests for this workflow.
    """
    workflow = Workflow(
        name=name,
        test_sample_type=test_sample_type,
        ground_truth_type=ground_truth_type,
        inference_type=inference_type,
    )

    test_case = type("TestCase", (TestCase,), {"workflow": workflow})
    test_suite = type("TestSuite", (TestSuite,), {"workflow": workflow, "_test_case_type": test_case})
    model = type("Model", (Model,), {"workflow": workflow})

    return workflow, cast(Type[TestCase], test_case), cast(Type[TestSuite], test_suite), cast(Type[Model], model)
