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
from kolena.classification.multiclass import Model
from kolena.classification.multiclass import MulticlassClassificationEvaluator
from kolena.classification.multiclass import TestSuite
from kolena.classification.multiclass.workflow import ThresholdConfiguration
from kolena.workflow import EvaluatorConfiguration
from kolena.workflow.test_run import test as base_test
from kolena.workflow.test_run import TestRun as BaseTestRun


class TestRun(BaseTestRun):
    """
    Convenience alias for :class:`kolena.workflow.test_run.TestRun`, configured for the
    `kolena.classification.multiclass` workflow and evaluator.

    Interface to run tests for a :class:`kolena.classification.multiclass.Model` on a set of
    :class:`kolena.classification.multiclass.TestSuite` suites.

    For a streamlined interface, see :meth:`kolena.classification.multiclass.test`.

    :param model: the model being tested.
    :param test_suite: the test suite on which to test the model.
    :param reset: overwrites existing inferences if set.
    """

    @validate_arguments(config=ValidatorConfig)
    def __init__(
        self,
        model: Model,
        test_suite: TestSuite,
        configurations: Optional[List[EvaluatorConfiguration]] = None,
        reset: bool = False,
    ):
        if configurations is None:
            configurations = [ThresholdConfiguration()]
        super().__init__(
            model=model,
            test_suite=test_suite,
            evaluator=MulticlassClassificationEvaluator,
            configurations=configurations,
            reset=reset,
        )


@validate_arguments(config=ValidatorConfig)
def test(
    model: Model,
    test_suite: TestSuite,
    configurations: Optional[List[EvaluatorConfiguration]] = None,
    reset: bool = False,
) -> None:
    """
    Convenience alias for :meth:`kolena.workflow.test_run.test` configured for the `kolena.classification.multiclass`
    workflow and evaluator.

    Tests the provided :class:`kolena.classification.multiclass.Model`` on the provided
    :class:`kolena.classification.multiclass.TestSuite` suite. Any tests already in progress for this model on this
    test suite is resumed.

    :param model: the model being tested, implementing the ``infer`` method.
    :param test_suite: the test suite on which to test the model.
    :param configurations: an optional list of configurations to use when running the evaluator. Defaults to selecting
        the max confidence label (with no thresholding) if unset.
    :param reset: overwrites existing inferences if set.
    """
    if configurations is None:
        configurations = [ThresholdConfiguration()]
    base_test(
        model=model,
        test_suite=test_suite,
        evaluator=MulticlassClassificationEvaluator,
        configurations=configurations,
        reset=reset,
    )
