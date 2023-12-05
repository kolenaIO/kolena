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
from typing import Tuple

import pytest

from kolena.workflow.annotation import LabeledPolygon
from kolena.workflow.annotation import ScoredClassificationLabel
from kolena.workflow.annotation import ScoredLabeledPolygon
from kolena.workflow.metrics import InferenceMatches
from kolena.workflow.metrics import MulticlassInferenceMatches

object_detection = pytest.importorskip("kolena._experimental.object_detection", reason="requires kolena[metrics] extra")
GroundTruth = object_detection.GroundTruth
Inference = object_detection.Inference
TestSample = object_detection.TestSample
ThresholdConfiguration = object_detection.ThresholdConfiguration
TestSampleMetrics = object_detection.TestSampleMetrics
TestSampleMetricsSingleClass = object_detection.TestSampleMetricsSingleClass

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
def test__object_detection__multiple_configurations__multiclass__polygon() -> None:
    polygon_gt = LabeledPolygon(points=[(0, 0), (10, 0), (10, 10), (0, 10)], label="a")
    polygon_inf = ScoredLabeledPolygon([(0, 0), (10, 0), (10.1, 10.1), (10, 10), (0, 10)], "b", 0.9)
    ground_truth = GroundTruth(bboxes=[polygon_gt])
    inference = Inference(
        bboxes=[polygon_inf],
    )

    config_one = ThresholdConfiguration(
        threshold_strategy=0.5,
        iou_threshold=0.5,
        min_confidence_score=0.001,
    )

    config_two = ThresholdConfiguration(
        threshold_strategy=0.3,
        iou_threshold=1,
        min_confidence_score=0,
    )

    evaluator = MulticlassObjectDetectionEvaluator()
    evaluator.compute_image_metrics(
        ground_truth=ground_truth,
        inference=inference,
        configuration=config_one,
        test_case_name="one_polygon",
    )
    evaluator.compute_image_metrics(
        ground_truth=ground_truth,
        inference=inference,
        configuration=config_two,
        test_case_name="two_polygon",
    )

    assert len(evaluator.matchings_by_test_case) == 2
    assert len(evaluator.matchings_by_test_case[config_one.display_name()]["one_polygon"]) == 1
    assert len(evaluator.matchings_by_test_case[config_two.display_name()]["two_polygon"]) == 1


@pytest.mark.metrics
def test__object_detection__multiple_configurations__single_class__polygon() -> None:
    polygon_gt = LabeledPolygon(points=[(0, 0), (10, 0), (10, 10), (0, 10)], label="a")
    polygon_inf = ScoredLabeledPolygon([(0, 0), (10, 0), (10.1, 10.1), (10, 10), (0, 10)], "b", 0.9)
    ground_truth = GroundTruth(bboxes=[polygon_gt])
    inference = Inference(
        bboxes=[polygon_inf],
    )

    config_one = ThresholdConfiguration(
        threshold_strategy=0.5,
        iou_threshold=0.5,
        min_confidence_score=0.001,
    )

    config_two = ThresholdConfiguration(
        threshold_strategy=0.3,
        iou_threshold=1,
        min_confidence_score=0,
    )

    evaluator = SingleClassObjectDetectionEvaluator()
    evaluator.compute_image_metrics(
        ground_truth=ground_truth,
        inference=inference,
        configuration=config_one,
        test_case_name="one_polygon",
    )
    evaluator.compute_image_metrics(
        ground_truth=ground_truth,
        inference=inference,
        configuration=config_two,
        test_case_name="two_polygon",
    )

    assert len(evaluator.matchings_by_test_case) == 2
    assert len(evaluator.matchings_by_test_case[config_one.display_name()]["one_polygon"]) == 1
    assert len(evaluator.matchings_by_test_case[config_two.display_name()]["two_polygon"]) == 1


@pytest.mark.metrics
def test__object_detection__evaluator__polygon() -> None:
    multiclass_matches: List[MulticlassInferenceMatches] = [
        MulticlassInferenceMatches(
            matched=[
                (
                    LabeledPolygon([(0, 0), (10, 10), (10, 0)], "a"),
                    ScoredLabeledPolygon([(0, 0), (10, 10), (10, 0)], "a", 0.9),
                ),
                (
                    LabeledPolygon([(0, 0), (10, 10), (0, 10)], "b"),
                    ScoredLabeledPolygon([(0, 0), (10, 10), (0, 10)], "b", 0.8),
                ),
            ],
            unmatched_gt=[],
            unmatched_inf=[],
        ),
        MulticlassInferenceMatches(
            matched=[],
            unmatched_gt=[
                (
                    LabeledPolygon([(1, 1), (7, 7), (7, 0)], "b"),
                    ScoredLabeledPolygon([(1, 1), (7, 7), (7, 0)], "a", 0.7),
                ),
            ],
            unmatched_inf=[
                ScoredLabeledPolygon([(0, 0), (5, 5), (10, 0)], "a", 0.7),
            ],
        ),
    ]

    single_class_matches = [
        InferenceMatches(
            matched=[
                (
                    LabeledPolygon([(0, 0), (10, 10), (10, 0)], "a"),
                    ScoredLabeledPolygon([(0, 0), (10, 10), (10, 0)], "a", 0.9),
                ),
                (
                    LabeledPolygon([(0, 0), (10, 10), (0, 10)], "a"),
                    ScoredLabeledPolygon([(0, 0), (10, 10), (0, 10)], "a", 0.8),
                ),
            ],
            unmatched_gt=[],
            unmatched_inf=[],
        ),
        InferenceMatches(
            matched=[],
            unmatched_gt=[
                LabeledPolygon([(1, 1), (7, 7), (7, 0)], "a"),
            ],
            unmatched_inf=[
                ScoredLabeledPolygon([(0, 0), (5, 5), (10, 0)], "a", 0.7),
            ],
        ),
    ]

    thresh = {"a": 0.85, "b": 0.85}

    eval = MulticlassObjectDetectionEvaluator()
    eval_single = SingleClassObjectDetectionEvaluator()

    assert eval.test_sample_metrics(multiclass_matches[0], thresh) == TestSampleMetrics(
        TP=[ScoredLabeledPolygon(points=[(0, 0), (10, 10), (10, 0)], label="a", score=0.9)],
        FP=[],
        FN=[LabeledPolygon(points=[(0, 0), (10, 10), (0, 10)], label="b")],
        Confused=[],
        count_TP=1,
        count_FP=0,
        count_FN=1,
        count_Confused=0,
        has_TP=True,
        has_FP=False,
        has_FN=True,
        has_Confused=False,
        ignored=False,
        max_confidence_above_t=0.9,
        min_confidence_above_t=0.9,
        thresholds=[ScoredClassificationLabel(label="a", score=0.85), ScoredClassificationLabel(label="b", score=0.85)],
    )

    assert eval.test_sample_metrics(multiclass_matches[1], thresh) == TestSampleMetrics(
        TP=[],
        FP=[],
        FN=[LabeledPolygon(points=[(1, 1), (7, 7), (7, 0)], label="b")],
        Confused=[],
        count_TP=0,
        count_FP=0,
        count_FN=1,
        count_Confused=0,
        has_TP=False,
        has_FP=False,
        has_FN=True,
        has_Confused=False,
        ignored=False,
        max_confidence_above_t=None,
        min_confidence_above_t=None,
        thresholds=[ScoredClassificationLabel(label="a", score=0.85)],
    )

    assert eval_single.test_sample_metrics_single_class(single_class_matches[0], 0.85) == TestSampleMetricsSingleClass(
        TP=[ScoredLabeledPolygon(points=[(0, 0), (10, 10), (10, 0)], label="a", score=0.9)],
        FP=[],
        FN=[LabeledPolygon(points=[(0, 0), (10, 10), (0, 10)], label="a")],
        count_TP=1,
        count_FP=0,
        count_FN=1,
        has_TP=True,
        has_FP=False,
        has_FN=True,
        ignored=False,
        max_confidence_above_t=0.9,
        min_confidence_above_t=0.9,
        thresholds=0.85,
    )
    assert eval_single.test_sample_metrics_single_class(single_class_matches[1], 0.85) == TestSampleMetricsSingleClass(
        TP=[],
        FP=[],
        FN=[LabeledPolygon(points=[(1, 1), (7, 7), (7, 0)], label="a")],
        count_TP=0,
        count_FP=0,
        count_FN=1,
        has_TP=False,
        has_FP=False,
        has_FN=True,
        ignored=False,
        max_confidence_above_t=None,
        min_confidence_above_t=None,
        thresholds=0.85,
    )


@pytest.mark.metrics
def test__object_detection__compute_and_cache_f1_optimal_thresholds__polygon() -> None:
    inferences: List[Tuple[TestSample, GroundTruth, Inference]] = [
        (
            TestSample(locator="s3://bucket/fake0.png"),
            GroundTruth(bboxes=[LabeledPolygon([(0, 0), (10, 10), (10, 0)], "a")]),
            Inference(bboxes=[ScoredLabeledPolygon([(0, 0), (10, 10), (10, 0)], "a", 0.9)]),
        ),
        (
            TestSample(locator="s3://bucket/fake1.png"),
            GroundTruth(
                bboxes=[
                    LabeledPolygon([(0, 30), (10, 40), (10, 30)], "a"),
                    LabeledPolygon([(30, 30), (10, 40), (10, 30)], "a"),
                    LabeledPolygon([(0, 30), (10, 40), (10, 30)], "a"),
                ],
            ),
            Inference(
                bboxes=[
                    ScoredLabeledPolygon([(0, 30), (10, 40), (10, 30)], "a", 0.9),
                    ScoredLabeledPolygon([(1, 30), (10, 40), (10, 30)], "a", 0.8),
                    ScoredLabeledPolygon([(30, 30), (10, 40), (10, 30)], "a", 0.7),
                    ScoredLabeledPolygon([(3, 30), (10, 40), (10, 30)], "a", 0.6),
                    ScoredLabeledPolygon([(4, 30), (10, 40), (10, 30)], "a", 0.5),
                    ScoredLabeledPolygon([(5, 30), (10, 40), (10, 30)], "a", 0.4),
                ],
            ),
        ),
    ]

    config = ThresholdConfiguration(min_confidence_score=0.45)
    eval_single = SingleClassObjectDetectionEvaluator()
    assert len(eval_single.threshold_cache) == 0
    eval_single.compute_and_cache_f1_optimal_thresholds(config, inferences)
    assert eval_single.threshold_cache[config.display_name()] == 0.7


@pytest.mark.metrics
def test__object_detection__compute_and_cache_f1_optimal_thresholds__polygon__multiclass() -> None:
    inferences: List[Tuple[TestSample, GroundTruth, Inference]] = [
        (
            TestSample(locator="s3://bucket/fake0.png"),
            GroundTruth(bboxes=[LabeledPolygon([(0, 0), (10, 10), (10, 0)], "a")]),
            Inference(bboxes=[ScoredLabeledPolygon([(0, 0), (10, 10), (10, 0)], "a", 0.9)]),
        ),
        (
            TestSample(locator="s3://bucket/fake1.png"),
            GroundTruth(
                bboxes=[
                    LabeledPolygon([(0, 30), (10, 40), (10, 30)], "a"),
                    LabeledPolygon([(30, 30), (10, 40), (10, 30)], "b"),
                    LabeledPolygon([(0, 30), (10, 40), (10, 30)], "a"),
                ],
            ),
            Inference(
                bboxes=[
                    ScoredLabeledPolygon([(0, 30), (10, 40), (10, 30)], "a", 0.9),
                    ScoredLabeledPolygon([(1, 30), (10, 40), (10, 30)], "a", 0.8),
                    ScoredLabeledPolygon([(30, 30), (10, 40), (10, 30)], "b", 0.7),
                    ScoredLabeledPolygon([(3, 30), (10, 40), (10, 30)], "a", 0.6),
                    ScoredLabeledPolygon([(4, 30), (10, 40), (10, 30)], "c", 0.5),
                    ScoredLabeledPolygon([(5, 30), (10, 40), (10, 30)], "a", 0.4),
                ],
            ),
        ),
    ]

    config = ThresholdConfiguration(min_confidence_score=0.45)
    eval = MulticlassObjectDetectionEvaluator()
    assert len(eval.threshold_cache) == 0
    eval.compute_and_cache_f1_optimal_thresholds(config, inferences)
    assert eval.threshold_cache[config.display_name()] == {"a": 0.9, "b": 0.7, "c": 0}
