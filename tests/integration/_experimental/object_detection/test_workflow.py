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

from kolena._experimental.object_detection import GroundTruth
from kolena._experimental.object_detection import Inference
from kolena._experimental.object_detection import Model
from kolena._experimental.object_detection import TestCase
from kolena._experimental.object_detection import TestSample
from kolena._experimental.object_detection import TestSuite
from kolena.workflow import test
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import ScoredLabeledBoundingBox
from tests.integration.helper import fake_locator
from tests.integration.helper import with_test_prefix


def test__multiclass__object__detection__smoke() -> None:
    name = with_test_prefix(f"{__file__} test__multiclass__object__detection__smoke")
    test_sample = TestSample(locator=fake_locator(0), metadata=dict(example="metadata", values=[1, 2, 3]))
    ground_truth = GroundTruth(bboxes=[LabeledBoundingBox(top_left=(0, 0), bottom_right=(1, 1), label="a")])
    test_case = TestCase(f"{name} test case", test_samples=[(test_sample, ground_truth)])
    test_suite = TestSuite(f"{name} test suite", test_cases=[test_case])

    def infer(_: TestSample) -> Inference:
        return Inference(
            inferences=[
                ScoredLabeledBoundingBox(top_left=(0, 0), bottom_right=(1, 1), label="b", score=random.random()),
                ScoredLabeledBoundingBox(top_left=(0, 0), bottom_right=(1.1, 1.1), label="a", score=random.random()),
                ScoredLabeledBoundingBox(top_left=(0, 0), bottom_right=(1.01, 1.01), label="a", score=0.99),
            ],
        )

    model = Model(f"{name} model", infer=infer)

    test(model, test_suite)
