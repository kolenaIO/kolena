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
from typing import Optional

from pydantic import validate_arguments

from kolena._utils.validators import ValidatorConfig
from kolena.classification.multiclass import evaluate_multiclass_classification
from kolena.classification.multiclass import Model
from kolena.classification.multiclass import TestSuite
from kolena.classification.multiclass.workflow import ThresholdConfiguration
from kolena.workflow.test_run import test as base_test


@validate_arguments(config=ValidatorConfig)
def test(
    model: Model,
    test_suite: TestSuite,
    configurations: Optional[List[ThresholdConfiguration]] = None,
    reset: bool = False,
) -> None:
    """
    Convenience alias for [`test`][kolena.workflow.test] configured for the pre-built Multiclass Classification
    workflow.

    Tests the provided [`Model`][kolena.classification.multiclass.Model] on the provided
    [`TestSuite`][kolena.classification.multiclass.TestSuite]. Any test already in progress for this model on this
    test suite is resumed.

    :param model: The model being tested, implementing the `infer` method.
    :param test_suite: The test suite on which to test the model.
    :param configurations: An optional list of configurations to use when running the evaluator. Defaults to selecting
        the max confidence label (with no thresholding) if unset.
    :param reset: Overwrites existing inferences if set.
    """
    if configurations is None:
        configurations = [ThresholdConfiguration()]
    base_test(
        model=model,
        test_suite=test_suite,
        evaluator=evaluate_multiclass_classification,
        configurations=configurations,
        reset=reset,
    )
