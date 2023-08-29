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

from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import ScoredLabeledBoundingBox

object_detection = pytest.importorskip("kolena._experimental.object_detection", reason="requires kolena[metrics] extra")
GroundTruth = object_detection.GroundTruth
Inference = object_detection.Inference
ThresholdStrategy = object_detection.ThresholdStrategy
ThresholdConfiguration = object_detection.ThresholdConfiguration

evaluator_multiclass = pytest.importorskip(
    "kolena._experimental.object_detection.evaluator_multiclass",
    reason="requires kolena[metrics] extra",
)
MulticlassObjectDetectionEvaluator = evaluator_multiclass.MulticlassObjectDetectionEvaluator

evaluator_single_class = pytest.importorskip(
    "kolena._experimental.object_detection.evaluator_single_class",
    reason="requires kolena[metrics] extra",
)
SingleClassObjectDetectionEvaluator = evaluator_single_class.SingleClassObjectDetectionEvaluator


@pytest.mark.metrics
def test__object_detection__multiple_configurations__multiclass() -> None:
    ground_truth = GroundTruth(bboxes=[LabeledBoundingBox(top_left=(0, 0), bottom_right=(1, 1), label="a")])
    inference = Inference(
        bboxes=[ScoredLabeledBoundingBox(top_left=(0, 0), bottom_right=(1, 1), label="b", score=random.random())],
    )

    config_one = ThresholdConfiguration(
        threshold_strategy=ThresholdStrategy.FIXED_05,
        iou_threshold=0.5,
        with_class_level_metrics=True,
        min_confidence_score=0.001,
    )

    config_two = ThresholdConfiguration(
        threshold_strategy=ThresholdStrategy.FIXED_03,
        iou_threshold=1,
        with_class_level_metrics=True,
        min_confidence_score=0.0,
    )

    evaluator = MulticlassObjectDetectionEvaluator()
    evaluator.compute_image_metrics(
        ground_truth=ground_truth,
        inference=inference,
        configuration=config_one,
        test_case_name="one",
    )
    evaluator.compute_image_metrics(
        ground_truth=ground_truth,
        inference=inference,
        configuration=config_two,
        test_case_name="two",
    )

    assert len(evaluator.matchings_by_test_case) == 2
    assert len(evaluator.matchings_by_test_case[config_one.display_name()]) == 1
    assert len(evaluator.matchings_by_test_case[config_one.display_name()]["one"]) == 1
    assert len(evaluator.matchings_by_test_case[config_two.display_name()]) == 1
    assert len(evaluator.matchings_by_test_case[config_two.display_name()]["two"]) == 1


@pytest.mark.metrics
def test__object_detection__multiple_configurations__single_class() -> None:
    ground_truth = GroundTruth(bboxes=[LabeledBoundingBox(top_left=(0, 0), bottom_right=(1, 1), label="a")])
    inference = Inference(
        bboxes=[ScoredLabeledBoundingBox(top_left=(0, 0), bottom_right=(1, 1), label="b", score=random.random())],
    )

    config_one = ThresholdConfiguration(
        threshold_strategy=ThresholdStrategy.FIXED_05,
        iou_threshold=0.5,
        with_class_level_metrics=False,
        min_confidence_score=0.001,
    )

    config_two = ThresholdConfiguration(
        threshold_strategy=ThresholdStrategy.FIXED_03,
        iou_threshold=1,
        with_class_level_metrics=False,
        min_confidence_score=0.0,
    )

    evaluator = SingleClassObjectDetectionEvaluator()
    evaluator.compute_image_metrics(
        ground_truth=ground_truth,
        inference=inference,
        configuration=config_one,
        test_case_name="one",
    )
    evaluator.compute_image_metrics(
        ground_truth=ground_truth,
        inference=inference,
        configuration=config_two,
        test_case_name="two",
    )

    assert len(evaluator.matchings_by_test_case) == 2
    assert len(evaluator.matchings_by_test_case[config_one.display_name()]) == 1
    assert len(evaluator.matchings_by_test_case[config_one.display_name()]["one"]) == 1
    assert len(evaluator.matchings_by_test_case[config_two.display_name()]) == 1
    assert len(evaluator.matchings_by_test_case[config_two.display_name()]["two"]) == 1
