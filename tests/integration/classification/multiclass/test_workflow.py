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
from typing import Type
from typing import Union

import pytest

from kolena.classification.multiclass import GroundTruth
from kolena.classification.multiclass import Inference
from kolena.classification.multiclass import InferenceLabel
from kolena.classification.multiclass import Model
from kolena.classification.multiclass import test
from kolena.classification.multiclass import TestCase
from kolena.classification.multiclass import TestSample
from kolena.classification.multiclass import TestSuite
from kolena.workflow.annotation import ClassificationLabel
from kolena.workflow.annotation import ScoredClassificationLabel
from tests.integration.helper import fake_locator
from tests.integration.helper import with_test_prefix


@pytest.mark.parametrize("inference_type", [ScoredClassificationLabel, InferenceLabel])
def test__multiclass_classification__smoke(
    inference_type: Union[Type[ScoredClassificationLabel], Type[InferenceLabel]],
) -> None:
    name = with_test_prefix(f"{__file__} test__multiclass_classification__smoke ({inference_type.__name__})")
    test_sample = TestSample(locator=fake_locator(0), metadata=dict(example="metadata", values=[1, 2, 3]))
    ground_truth = GroundTruth(classification=ClassificationLabel(label="example"))
    test_case = TestCase(f"{name} test case", test_samples=[(test_sample, ground_truth)])
    test_suite = TestSuite(f"{name} test suite", test_cases=[test_case])

    def infer(_: TestSample) -> Inference:
        score = random.random()
        inferences = [("example", 1 - score), ("another", score / 2), ("third", score / 2)]
        return Inference(inferences=[inference_type(label, conf) for label, conf in inferences])

    model = Model(f"{name} model", infer=infer)
    test(model, test_suite)  # TODO: add detailed unit tests for MulticlassClassificationEvaluator
