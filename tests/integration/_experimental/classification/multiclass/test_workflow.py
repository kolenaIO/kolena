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
import dataclasses
import random
from typing import Optional
from typing import Type

import pydantic
import pytest

from kolena._experimental.classification.multiclass import GroundTruth
from kolena._experimental.classification.multiclass import Inference
from kolena._experimental.classification.multiclass import Model
from kolena._experimental.classification.multiclass import test
from kolena._experimental.classification.multiclass import TestCase
from kolena._experimental.classification.multiclass import TestSuite
from kolena.workflow import Image
from kolena.workflow import Metadata
from kolena.workflow.annotation import ClassificationLabel
from kolena.workflow.annotation import ScoredClassificationLabel
from tests.integration.helper import fake_locator
from tests.integration.helper import with_test_prefix


def test__multiclass_classification__smoke() -> None:
    name = with_test_prefix(f"{__file__} test__multiclass_classification__smoke")
    test_sample = Image(locator=fake_locator(0))
    ground_truth = GroundTruth(classification=ClassificationLabel(label="example"))
    test_case = TestCase(f"{name} test case", test_samples=[(test_sample, ground_truth)])
    test_suite = TestSuite(f"{name} test suite", test_cases=[test_case])

    def infer(_: Image) -> Inference:
        score = random.random()
        inferences = [("example", 1 - score), ("another", score / 2), ("third", score / 2)]
        return Inference(inferences=[ScoredClassificationLabel(label=label, score=conf) for label, conf in inferences])

    model = Model(f"{name} model", infer=infer)
    test(model, test_suite)  # TODO: add detailed unit tests for evaluate_multiclass_classification


@pytest.mark.parametrize("dataclass_type", [dataclasses.dataclass, pydantic.dataclasses.dataclass])
def test__multiclass_classification__extends_image(dataclass_type: Type) -> None:
    @dataclass_type(frozen=True)
    class ExtendsImage(Image):
        example: str
        optional: Optional[int] = None
        metadata: Metadata = dataclasses.field(default_factory=dict)

    test_sample = ExtendsImage(locator="s3://test-bucket/example.png", example="yes", metadata=dict(example=1))
    ground_truth = GroundTruth(classification=ClassificationLabel(label="example"))

    # any test sample extending kolena.workflow.Image should pass validation
    TestCase(f"{__file__} test__multiclass_classification__extends_image", test_samples=[(test_sample, ground_truth)])
