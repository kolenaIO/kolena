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

import pytest

from kolena.workflow import test
from kolena.workflow.annotation import LabeledPolygon
from kolena.workflow.annotation import ScoredLabeledPolygon
from tests.integration.helper import fake_locator
from tests.integration.helper import with_test_prefix


instance_segmentation = pytest.importorskip(
    "kolena._experimental.instance_segmentation",
    reason="requires kolena[metrics] extra",
)
GroundTruth = instance_segmentation.GroundTruth
Inference = instance_segmentation.Inference
Model = instance_segmentation.Model
TestCase = instance_segmentation.TestCase
TestSample = instance_segmentation.TestSample
TestSuite = instance_segmentation.TestSuite
EvaluatorConfiguration = instance_segmentation.EvaluatorConfiguration
InstanceSegmentationEvaluator = instance_segmentation.InstanceSegmentationEvaluator


@pytest.mark.metrics
def test__instance_segmentation__smoke() -> None:
    name = with_test_prefix(f"{__file__} test__multiclass__object_detection__smoke")
    test_sample = TestSample(locator=fake_locator(0), metadata=dict(example="metadata", values=[1, 2, 3]))
    ground_truth = GroundTruth(polygons=[LabeledPolygon([(0, 1), (0, 2), (2, 1), (2, 2)], label="a")])
    test_case = TestCase(f"{name} test case", test_samples=[(test_sample, ground_truth)])
    test_suite = TestSuite(f"{name} test suite", test_cases=[test_case])

    def infer(_: TestSample) -> Inference:
        return Inference(
            polygons=[
                ScoredLabeledPolygon(points=[(0, 0), (0, 1), (1, 0), (1, 1)], label="b", score=random.random()),
                ScoredLabeledPolygon(points=[(0, 0), (0, 1.1), (1.1, 0), (1.1, 1.1)], label="a", score=random.random()),
                ScoredLabeledPolygon(points=[(0, 0), (0, 1.01), (1.01, 0), (1.01, 1.01)], label="a", score=0.99),
            ],
        )

    evaluator = InstanceSegmentationEvaluator(
        configurations=[EvaluatorConfiguration()],
    )

    model = Model(f"{name} model", infer=infer)
    test(model, test_suite, evaluator, reset=True)
