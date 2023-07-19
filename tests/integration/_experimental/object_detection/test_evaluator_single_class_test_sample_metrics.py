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
from typing import Dict
from typing import List
from typing import Tuple

import pytest

from kolena._experimental.object_detection import GroundTruth
from kolena._experimental.object_detection import Inference
from kolena._experimental.object_detection import ObjectDetectionEvaluator
from kolena._experimental.object_detection import TestCase
from kolena._experimental.object_detection import TestSample
from kolena._experimental.object_detection import ThresholdConfiguration
from kolena._experimental.object_detection import ThresholdStrategy
from kolena._experimental.object_detection.workflow import TestSampleMetricsSingleClass
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import ScoredLabeledBoundingBox
from tests.integration.helper import fake_locator
from tests.integration.helper import with_test_prefix

TEST_CASE = TestCase(with_test_prefix("test_evaluator_single_class"), reset=True)


TEST_DATA: Dict[str, List[Tuple[TestSample, GroundTruth, Inference]]] = {
    "nothing": [
        (
            TestSample(locator=fake_locator(1, "OD")),
            GroundTruth(
                bboxes=[],
            ),
            Inference(
                bboxes=[],
            ),
        ),
    ],
    "no inferences": [
        (
            TestSample(locator=fake_locator(2, "OD")),
            GroundTruth(
                bboxes=[LabeledBoundingBox((3, 3), (4, 4), "b")],
            ),
            Inference(
                bboxes=[],
            ),
        ),
    ],
    "no ground truths": [
        (
            TestSample(locator=fake_locator(3, "OD")),
            GroundTruth(
                bboxes=[],
            ),
            Inference(
                bboxes=[ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 1)],
            ),
        ),
    ],
    "iou=1 and different labels and max confidence": [
        (
            TestSample(locator=fake_locator(4, "OD")),
            GroundTruth(
                bboxes=[LabeledBoundingBox((1, 1), (2, 2), "a")],
            ),
            Inference(
                bboxes=[ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 1)],
            ),
        ),
    ],
    "iou=0 and same labels": [
        (
            TestSample(locator=fake_locator(5, "OD")),
            GroundTruth(
                bboxes=[LabeledBoundingBox((3, 3), (4, 4), "b")],
            ),
            Inference(
                bboxes=[ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 1)],
            ),
        ),
    ],
    "iou=0.33 and same labels but 0 confidence": [
        (
            TestSample(locator=fake_locator(6, "OD")),
            GroundTruth(
                bboxes=[LabeledBoundingBox((1, 1), (4, 4), "b")],
            ),
            Inference(
                bboxes=[ScoredLabeledBoundingBox((2, 2), (5, 5), "b", 0)],
            ),
        ),
    ],
    "iou=0.33 and same labels but 0.5 confidence": [
        (
            TestSample(locator=fake_locator(7, "OD")),
            GroundTruth(
                bboxes=[LabeledBoundingBox((1, 1), (4, 4), "b")],
            ),
            Inference(
                bboxes=[ScoredLabeledBoundingBox((2, 2), (5, 5), "b", 0.5)],
            ),
        ),
    ],
    "iou=0.33 and same labels but 0.99 confidence": [
        (
            TestSample(locator=fake_locator(8, "OD")),
            GroundTruth(
                bboxes=[LabeledBoundingBox((1, 1), (4, 4), "b")],
            ),
            Inference(
                bboxes=[ScoredLabeledBoundingBox((2, 2), (5, 5), "b", 0.99)],
            ),
        ),
    ],
    "iou=0.5 and same labels but 0 confidence": [
        (
            TestSample(locator=fake_locator(9, "OD")),
            GroundTruth(
                bboxes=[LabeledBoundingBox((1, 1), (4, 4), "b")],
            ),
            Inference(
                bboxes=[ScoredLabeledBoundingBox((2, 1), (5, 4), "b", 0)],
            ),
        ),
    ],
    "iou=0.5 and same labels but 0.5 confidence": [
        (
            TestSample(locator=fake_locator(10, "OD")),
            GroundTruth(
                bboxes=[LabeledBoundingBox((1, 1), (4, 4), "b")],
            ),
            Inference(
                bboxes=[ScoredLabeledBoundingBox((2, 1), (5, 4), "b", 0.5)],
            ),
        ),
    ],
    "iou=0.5 and same labels but 0.99 confidence": [
        (
            TestSample(locator=fake_locator(11, "OD")),
            GroundTruth(
                bboxes=[LabeledBoundingBox((1, 1), (4, 4), "b")],
            ),
            Inference(
                bboxes=[ScoredLabeledBoundingBox((2, 1), (5, 4), "b", 0.99)],
            ),
        ),
    ],
    "multiple bboxes in an image, perfect match": [
        (
            TestSample(locator=fake_locator(12, "OD")),
            GroundTruth(
                bboxes=[
                    LabeledBoundingBox((1, 1), (2, 2), "a"),
                    LabeledBoundingBox((3, 3), (4, 4), "a"),
                    LabeledBoundingBox((5, 5), (6, 6), "a"),
                    LabeledBoundingBox((7, 7), (8, 8), "a"),
                ],
            ),
            Inference(
                bboxes=[
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.9),
                    ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.9),
                    ScoredLabeledBoundingBox((5, 5), (6, 6), "a", 0.9),
                    ScoredLabeledBoundingBox((7, 7), (8, 8), "a", 0.9),
                ],
            ),
        ),
    ],
    "multiple bboxes in an image, varied iou": [
        (
            TestSample(locator=fake_locator(13, "OD")),
            GroundTruth(
                bboxes=[
                    LabeledBoundingBox((1, 1), (2, 2), "a"),
                    LabeledBoundingBox((3, 3), (4, 4), "a"),
                    LabeledBoundingBox((5, 5), (6, 6), "a"),
                    LabeledBoundingBox((7, 7), (8, 8), "a"),
                ],
            ),
            Inference(
                bboxes=[
                    ScoredLabeledBoundingBox((1.1, 1), (2.1, 2), "a", 0.9),
                    ScoredLabeledBoundingBox((3.3, 3), (4.3, 4), "a", 0.9),
                    ScoredLabeledBoundingBox((5.5, 5), (6.5, 6), "a", 0.9),
                    ScoredLabeledBoundingBox((7.7, 7), (8.7, 8), "a", 0.9),
                ],
            ),
        ),
    ],
    "multiple bboxes in an image, varied confidence": [
        (
            TestSample(locator=fake_locator(14, "OD")),
            GroundTruth(
                bboxes=[
                    LabeledBoundingBox((1, 1), (2, 2), "a"),
                    LabeledBoundingBox((3, 3), (4, 4), "a"),
                    LabeledBoundingBox((5, 5), (6, 6), "a"),
                    LabeledBoundingBox((7, 7), (8, 8), "a"),
                ],
            ),
            Inference(
                bboxes=[
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.6),
                    ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.5),
                    ScoredLabeledBoundingBox((5, 5), (6, 6), "a", 0.4),
                    ScoredLabeledBoundingBox((7, 7), (8, 8), "a", 0.01),
                ],
            ),
        ),
    ],
    "multiple bboxes in an image, many inferences": [
        (
            TestSample(locator=fake_locator(15, "OD")),
            GroundTruth(
                bboxes=[
                    LabeledBoundingBox((1, 1), (2, 2), "a"),
                    LabeledBoundingBox((3, 3), (4, 4), "a"),
                    LabeledBoundingBox((5, 5), (6, 6), "a"),
                    LabeledBoundingBox((7, 7), (8, 8), "a"),
                ],
            ),
            Inference(
                bboxes=[
                    ScoredLabeledBoundingBox((0, 0), (1, 1), "a", 0.99),
                    ScoredLabeledBoundingBox((0, 0), (1, 1), "a", 0.99),
                    ScoredLabeledBoundingBox((0, 0), (1, 1), "a", 0.99),
                    ScoredLabeledBoundingBox((0, 0), (1, 1), "a", 0.99),
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.99),
                    ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.99),
                    ScoredLabeledBoundingBox((5, 5), (6, 6), "a", 0.99),
                    ScoredLabeledBoundingBox((7, 7), (8, 8), "a", 0.99),
                ],
            ),
        ),
    ],
    "multiple bboxes in an image, too few inferences": [
        (
            TestSample(locator=fake_locator(16, "OD")),
            GroundTruth(
                bboxes=[
                    LabeledBoundingBox((1, 1), (2, 2), "a"),
                    LabeledBoundingBox((3, 3), (4, 4), "a"),
                    LabeledBoundingBox((5, 5), (6, 6), "a"),
                    LabeledBoundingBox((7, 7), (8, 8), "a"),
                ],
            ),
            Inference(
                bboxes=[
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.99),
                    ScoredLabeledBoundingBox((7, 7), (8, 8), "a", 0.99),
                ],
            ),
        ),
    ],
    "multiple bboxes in an image, suboptimal infs": [
        (
            TestSample(locator=fake_locator(17, "OD")),
            GroundTruth(
                bboxes=[
                    LabeledBoundingBox((1, 1), (2, 2), "a"),
                    LabeledBoundingBox((3, 3), (4, 4), "a"),
                    LabeledBoundingBox((5, 5), (6, 6), "a"),
                    LabeledBoundingBox((7, 7), (8, 8), "a"),
                ],
            ),
            Inference(
                bboxes=[
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.001),
                    ScoredLabeledBoundingBox((5, 5), (6, 6), "a", 0.9),
                    ScoredLabeledBoundingBox((7, 7), (9, 9), "a", 0.9),
                ],
            ),
        ),
    ],
    "multiple bboxes in an image, ignored matches": [
        (
            TestSample(locator=fake_locator(18, "OD")),
            GroundTruth(
                bboxes=[
                    LabeledBoundingBox((1, 1), (2, 2), "a"),
                    LabeledBoundingBox((3, 3), (4, 4), "a"),
                ],
                ignored_bboxes=[
                    LabeledBoundingBox((5, 5), (6, 6), "a"),
                    LabeledBoundingBox((7, 7), (8, 8), "a"),
                ],
            ),
            Inference(
                bboxes=[
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.9),
                    ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.9),
                    ScoredLabeledBoundingBox((5, 5), (6, 6), "a", 0.01),
                    ScoredLabeledBoundingBox((7, 7), (8, 8), "a", 0.9),
                ],
            ),
        ),
    ],
}


TEST_CONFIGURATIONS: Dict[str, ThresholdConfiguration] = {
    "Threshold: Fixed(0.3), IoU: 0.3, confidence ≥ 0.0": ThresholdConfiguration(
        threshold_strategy=ThresholdStrategy.FIXED_03,
        iou_threshold=0.3,
        min_confidence_score=0.0,
        with_class_level_metrics=False,
    ),
    "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.0": ThresholdConfiguration(
        threshold_strategy=ThresholdStrategy.FIXED_05,
        iou_threshold=0.5,
        min_confidence_score=0.0,
        with_class_level_metrics=False,
    ),
    "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.3": ThresholdConfiguration(
        threshold_strategy=ThresholdStrategy.FIXED_05,
        iou_threshold=0.5,
        min_confidence_score=0.3,
        with_class_level_metrics=False,
    ),
    "Threshold: F1-Optimal, IoU: 0.5, confidence ≥ 0.1": ThresholdConfiguration(
        threshold_strategy=ThresholdStrategy.F1_OPTIMAL,
        iou_threshold=0.5,
        min_confidence_score=0.1,
        with_class_level_metrics=False,
    ),
}


TEST_PARAMS = [(config, name) for name in TEST_DATA.keys() for config in TEST_CONFIGURATIONS.keys()]


# evaluator_configuration -> test_name -> test_sample_metrics
EXPECTED_COMPUTE_TEST_SAMPLE_METRICS: Dict[str, Dict[str, List[Tuple[TestSample, TestSampleMetricsSingleClass]]]] = {
    "Threshold: Fixed(0.3), IoU: 0.3, confidence ≥ 0.0": {
        "nothing": [
            (
                TestSample(locator=fake_locator(1, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[],
                    FN=[],
                    count_TP=0,
                    count_FP=0,
                    count_FN=0,
                    has_TP=False,
                    has_FP=False,
                    has_FN=False,
                    max_confidence_above_t=None,
                    min_confidence_above_t=None,
                    thresholds=0.3,
                ),
            ),
        ],
        "no inferences": [
            (
                TestSample(locator=fake_locator(2, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[],
                    FN=[LabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="b")],
                    count_TP=0,
                    count_FP=0,
                    count_FN=1,
                    has_TP=False,
                    has_FP=False,
                    has_FN=True,
                    max_confidence_above_t=None,
                    min_confidence_above_t=None,
                    thresholds=0.3,
                ),
            ),
        ],
        "no ground truths": [
            (
                TestSample(locator=fake_locator(3, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="b", score=1.0)],
                    FN=[],
                    count_TP=0,
                    count_FP=1,
                    count_FN=0,
                    has_TP=False,
                    has_FP=True,
                    has_FN=False,
                    max_confidence_above_t=1.0,
                    min_confidence_above_t=1.0,
                    thresholds=0.3,
                ),
            ),
        ],
        "iou=1 and different labels and max confidence": [
            (
                TestSample(locator=fake_locator(4, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="b", score=1.0)],
                    FP=[],
                    FN=[],
                    count_TP=1,
                    count_FP=0,
                    count_FN=0,
                    has_TP=True,
                    has_FP=False,
                    has_FN=False,
                    max_confidence_above_t=1.0,
                    min_confidence_above_t=1.0,
                    thresholds=0.3,
                ),
            ),
        ],
        "iou=0 and same labels": [
            (
                TestSample(locator=fake_locator(5, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="b", score=1.0)],
                    FN=[LabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="b")],
                    count_TP=0,
                    count_FP=1,
                    count_FN=1,
                    has_TP=False,
                    has_FP=True,
                    has_FN=True,
                    max_confidence_above_t=1.0,
                    min_confidence_above_t=1.0,
                    thresholds=0.3,
                ),
            ),
        ],
        "iou=0.33 and same labels but 0 confidence": [
            (
                TestSample(locator=fake_locator(6, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[],
                    FN=[LabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(4.0, 4.0), label="b")],
                    count_TP=0,
                    count_FP=0,
                    count_FN=1,
                    has_TP=False,
                    has_FP=False,
                    has_FN=True,
                    max_confidence_above_t=None,
                    min_confidence_above_t=None,
                    thresholds=0.3,
                ),
            ),
        ],
        "iou=0.33 and same labels but 0.5 confidence": [
            (
                TestSample(locator=fake_locator(7, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[ScoredLabeledBoundingBox(top_left=(2.0, 2.0), bottom_right=(5.0, 5.0), label="b", score=0.5)],
                    FN=[LabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(4.0, 4.0), label="b")],
                    count_TP=0,
                    count_FP=1,
                    count_FN=1,
                    has_TP=False,
                    has_FP=True,
                    has_FN=True,
                    max_confidence_above_t=0.5,
                    min_confidence_above_t=0.5,
                    thresholds=0.3,
                ),
            ),
        ],
        "iou=0.33 and same labels but 0.99 confidence": [
            (
                TestSample(locator=fake_locator(8, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[ScoredLabeledBoundingBox(top_left=(2.0, 2.0), bottom_right=(5.0, 5.0), label="b", score=0.99)],
                    FN=[LabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(4.0, 4.0), label="b")],
                    count_TP=0,
                    count_FP=1,
                    count_FN=1,
                    has_TP=False,
                    has_FP=True,
                    has_FN=True,
                    max_confidence_above_t=0.99,
                    min_confidence_above_t=0.99,
                    thresholds=0.3,
                ),
            ),
        ],
        "iou=0.5 and same labels but 0 confidence": [
            (
                TestSample(locator=fake_locator(9, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[],
                    FN=[LabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(4.0, 4.0), label="b")],
                    count_TP=0,
                    count_FP=0,
                    count_FN=1,
                    has_TP=False,
                    has_FP=False,
                    has_FN=True,
                    max_confidence_above_t=None,
                    min_confidence_above_t=None,
                    thresholds=0.3,
                ),
            ),
        ],
        "iou=0.5 and same labels but 0.5 confidence": [
            (
                TestSample(locator=fake_locator(10, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[ScoredLabeledBoundingBox(top_left=(2.0, 1.0), bottom_right=(5.0, 4.0), label="b", score=0.5)],
                    FP=[],
                    FN=[],
                    count_TP=1,
                    count_FP=0,
                    count_FN=0,
                    has_TP=True,
                    has_FP=False,
                    has_FN=False,
                    max_confidence_above_t=0.5,
                    min_confidence_above_t=0.5,
                    thresholds=0.3,
                ),
            ),
        ],
        "iou=0.5 and same labels but 0.99 confidence": [
            (
                TestSample(locator=fake_locator(11, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[ScoredLabeledBoundingBox(top_left=(2.0, 1.0), bottom_right=(5.0, 4.0), label="b", score=0.99)],
                    FP=[],
                    FN=[],
                    count_TP=1,
                    count_FP=0,
                    count_FN=0,
                    has_TP=True,
                    has_FP=False,
                    has_FN=False,
                    max_confidence_above_t=0.99,
                    min_confidence_above_t=0.99,
                    thresholds=0.3,
                ),
            ),
        ],
        "multiple bboxes in an image, perfect match": [
            (
                TestSample(locator=fake_locator(12, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[
                        ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="a", score=0.9),
                        ScoredLabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="a", score=0.9),
                        ScoredLabeledBoundingBox(top_left=(5.0, 5.0), bottom_right=(6.0, 6.0), label="a", score=0.9),
                        ScoredLabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(8.0, 8.0), label="a", score=0.9),
                    ],
                    FP=[],
                    FN=[],
                    count_TP=4,
                    count_FP=0,
                    count_FN=0,
                    has_TP=True,
                    has_FP=False,
                    has_FN=False,
                    max_confidence_above_t=0.9,
                    min_confidence_above_t=0.9,
                    thresholds=0.3,
                ),
            ),
        ],
        "multiple bboxes in an image, varied iou": [
            (
                TestSample(locator=fake_locator(13, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[
                        ScoredLabeledBoundingBox(top_left=(1.1, 1.0), bottom_right=(2.1, 2.0), label="a", score=0.9),
                        ScoredLabeledBoundingBox(top_left=(3.3, 3.0), bottom_right=(4.3, 4.0), label="a", score=0.9),
                        ScoredLabeledBoundingBox(top_left=(5.5, 5.0), bottom_right=(6.5, 6.0), label="a", score=0.9),
                    ],
                    FP=[ScoredLabeledBoundingBox(top_left=(7.7, 7.0), bottom_right=(8.7, 8.0), label="a", score=0.9)],
                    FN=[LabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(8.0, 8.0), label="a")],
                    count_TP=3,
                    count_FP=1,
                    count_FN=1,
                    has_TP=True,
                    has_FP=True,
                    has_FN=True,
                    max_confidence_above_t=0.9,
                    min_confidence_above_t=0.9,
                    thresholds=0.3,
                ),
            ),
        ],
        "multiple bboxes in an image, varied confidence": [
            (
                TestSample(locator=fake_locator(14, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[
                        ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="a", score=0.6),
                        ScoredLabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="a", score=0.5),
                        ScoredLabeledBoundingBox(top_left=(5.0, 5.0), bottom_right=(6.0, 6.0), label="a", score=0.4),
                    ],
                    FP=[],
                    FN=[LabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(8.0, 8.0), label="a")],
                    count_TP=3,
                    count_FP=0,
                    count_FN=1,
                    has_TP=True,
                    has_FP=False,
                    has_FN=True,
                    max_confidence_above_t=0.6,
                    min_confidence_above_t=0.4,
                    thresholds=0.3,
                ),
            ),
        ],
        "multiple bboxes in an image, many inferences": [
            (
                TestSample(locator=fake_locator(15, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[
                        ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(5.0, 5.0), bottom_right=(6.0, 6.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(8.0, 8.0), label="a", score=0.99),
                    ],
                    FP=[
                        ScoredLabeledBoundingBox(top_left=(0.0, 0.0), bottom_right=(1.0, 1.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(0.0, 0.0), bottom_right=(1.0, 1.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(0.0, 0.0), bottom_right=(1.0, 1.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(0.0, 0.0), bottom_right=(1.0, 1.0), label="a", score=0.99),
                    ],
                    FN=[],
                    count_TP=4,
                    count_FP=4,
                    count_FN=0,
                    has_TP=True,
                    has_FP=True,
                    has_FN=False,
                    max_confidence_above_t=0.99,
                    min_confidence_above_t=0.99,
                    thresholds=0.3,
                ),
            ),
        ],
        "multiple bboxes in an image, too few inferences": [
            (
                TestSample(locator=fake_locator(16, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[
                        ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(8.0, 8.0), label="a", score=0.99),
                    ],
                    FP=[],
                    FN=[
                        LabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="a"),
                        LabeledBoundingBox(top_left=(5.0, 5.0), bottom_right=(6.0, 6.0), label="a"),
                    ],
                    count_TP=2,
                    count_FP=0,
                    count_FN=2,
                    has_TP=True,
                    has_FP=False,
                    has_FN=True,
                    max_confidence_above_t=0.99,
                    min_confidence_above_t=0.99,
                    thresholds=0.3,
                ),
            ),
        ],
        "multiple bboxes in an image, suboptimal infs": [
            (
                TestSample(locator=fake_locator(17, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[ScoredLabeledBoundingBox(top_left=(5.0, 5.0), bottom_right=(6.0, 6.0), label="a", score=0.9)],
                    FP=[ScoredLabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(9.0, 9.0), label="a", score=0.9)],
                    FN=[
                        LabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="a"),
                        LabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(8.0, 8.0), label="a"),
                        LabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="a"),
                    ],
                    count_TP=1,
                    count_FP=1,
                    count_FN=3,
                    has_TP=True,
                    has_FP=True,
                    has_FN=True,
                    max_confidence_above_t=0.9,
                    min_confidence_above_t=0.9,
                    thresholds=0.3,
                ),
            ),
        ],
        "multiple bboxes in an image, ignored matches": [
            (
                TestSample(locator=fake_locator(18, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[
                        ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="a", score=0.9),
                        ScoredLabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="a", score=0.9),
                    ],
                    FP=[],
                    FN=[],
                    count_TP=2,
                    count_FP=0,
                    count_FN=0,
                    has_TP=True,
                    has_FP=False,
                    has_FN=False,
                    max_confidence_above_t=0.9,
                    min_confidence_above_t=0.9,
                    thresholds=0.3,
                ),
            ),
        ],
    },
    "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.0": {
        "nothing": [
            (
                TestSample(locator=fake_locator(1, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[],
                    FN=[],
                    count_TP=0,
                    count_FP=0,
                    count_FN=0,
                    has_TP=False,
                    has_FP=False,
                    has_FN=False,
                    max_confidence_above_t=None,
                    min_confidence_above_t=None,
                    thresholds=0.5,
                ),
            ),
        ],
        "no inferences": [
            (
                TestSample(locator=fake_locator(2, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[],
                    FN=[LabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="b")],
                    count_TP=0,
                    count_FP=0,
                    count_FN=1,
                    has_TP=False,
                    has_FP=False,
                    has_FN=True,
                    max_confidence_above_t=None,
                    min_confidence_above_t=None,
                    thresholds=0.5,
                ),
            ),
        ],
        "no ground truths": [
            (
                TestSample(locator=fake_locator(3, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="b", score=1.0)],
                    FN=[],
                    count_TP=0,
                    count_FP=1,
                    count_FN=0,
                    has_TP=False,
                    has_FP=True,
                    has_FN=False,
                    max_confidence_above_t=1.0,
                    min_confidence_above_t=1.0,
                    thresholds=0.5,
                ),
            ),
        ],
        "iou=1 and different labels and max confidence": [
            (
                TestSample(locator=fake_locator(4, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="b", score=1.0)],
                    FP=[],
                    FN=[],
                    count_TP=1,
                    count_FP=0,
                    count_FN=0,
                    has_TP=True,
                    has_FP=False,
                    has_FN=False,
                    max_confidence_above_t=1.0,
                    min_confidence_above_t=1.0,
                    thresholds=0.5,
                ),
            ),
        ],
        "iou=0 and same labels": [
            (
                TestSample(locator=fake_locator(5, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="b", score=1.0)],
                    FN=[LabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="b")],
                    count_TP=0,
                    count_FP=1,
                    count_FN=1,
                    has_TP=False,
                    has_FP=True,
                    has_FN=True,
                    max_confidence_above_t=1.0,
                    min_confidence_above_t=1.0,
                    thresholds=0.5,
                ),
            ),
        ],
        "iou=0.33 and same labels but 0 confidence": [
            (
                TestSample(locator=fake_locator(6, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[],
                    FN=[LabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(4.0, 4.0), label="b")],
                    count_TP=0,
                    count_FP=0,
                    count_FN=1,
                    has_TP=False,
                    has_FP=False,
                    has_FN=True,
                    max_confidence_above_t=None,
                    min_confidence_above_t=None,
                    thresholds=0.5,
                ),
            ),
        ],
        "iou=0.33 and same labels but 0.5 confidence": [
            (
                TestSample(locator=fake_locator(7, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[ScoredLabeledBoundingBox(top_left=(2.0, 2.0), bottom_right=(5.0, 5.0), label="b", score=0.5)],
                    FN=[LabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(4.0, 4.0), label="b")],
                    count_TP=0,
                    count_FP=1,
                    count_FN=1,
                    has_TP=False,
                    has_FP=True,
                    has_FN=True,
                    max_confidence_above_t=0.5,
                    min_confidence_above_t=0.5,
                    thresholds=0.5,
                ),
            ),
        ],
        "iou=0.33 and same labels but 0.99 confidence": [
            (
                TestSample(locator=fake_locator(8, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[ScoredLabeledBoundingBox(top_left=(2.0, 2.0), bottom_right=(5.0, 5.0), label="b", score=0.99)],
                    FN=[LabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(4.0, 4.0), label="b")],
                    count_TP=0,
                    count_FP=1,
                    count_FN=1,
                    has_TP=False,
                    has_FP=True,
                    has_FN=True,
                    max_confidence_above_t=0.99,
                    min_confidence_above_t=0.99,
                    thresholds=0.5,
                ),
            ),
        ],
        "iou=0.5 and same labels but 0 confidence": [
            (
                TestSample(locator=fake_locator(9, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[],
                    FN=[LabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(4.0, 4.0), label="b")],
                    count_TP=0,
                    count_FP=0,
                    count_FN=1,
                    has_TP=False,
                    has_FP=False,
                    has_FN=True,
                    max_confidence_above_t=None,
                    min_confidence_above_t=None,
                    thresholds=0.5,
                ),
            ),
        ],
        "iou=0.5 and same labels but 0.5 confidence": [
            (
                TestSample(locator=fake_locator(10, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[ScoredLabeledBoundingBox(top_left=(2.0, 1.0), bottom_right=(5.0, 4.0), label="b", score=0.5)],
                    FP=[],
                    FN=[],
                    count_TP=1,
                    count_FP=0,
                    count_FN=0,
                    has_TP=True,
                    has_FP=False,
                    has_FN=False,
                    max_confidence_above_t=0.5,
                    min_confidence_above_t=0.5,
                    thresholds=0.5,
                ),
            ),
        ],
        "iou=0.5 and same labels but 0.99 confidence": [
            (
                TestSample(locator=fake_locator(11, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[ScoredLabeledBoundingBox(top_left=(2.0, 1.0), bottom_right=(5.0, 4.0), label="b", score=0.99)],
                    FP=[],
                    FN=[],
                    count_TP=1,
                    count_FP=0,
                    count_FN=0,
                    has_TP=True,
                    has_FP=False,
                    has_FN=False,
                    max_confidence_above_t=0.99,
                    min_confidence_above_t=0.99,
                    thresholds=0.5,
                ),
            ),
        ],
        "multiple bboxes in an image, perfect match": [
            (
                TestSample(locator=fake_locator(12, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[
                        ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="a", score=0.9),
                        ScoredLabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="a", score=0.9),
                        ScoredLabeledBoundingBox(top_left=(5.0, 5.0), bottom_right=(6.0, 6.0), label="a", score=0.9),
                        ScoredLabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(8.0, 8.0), label="a", score=0.9),
                    ],
                    FP=[],
                    FN=[],
                    count_TP=4,
                    count_FP=0,
                    count_FN=0,
                    has_TP=True,
                    has_FP=False,
                    has_FN=False,
                    max_confidence_above_t=0.9,
                    min_confidence_above_t=0.9,
                    thresholds=0.5,
                ),
            ),
        ],
        "multiple bboxes in an image, varied iou": [
            (
                TestSample(locator=fake_locator(13, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[
                        ScoredLabeledBoundingBox(top_left=(1.1, 1.0), bottom_right=(2.1, 2.0), label="a", score=0.9),
                        ScoredLabeledBoundingBox(top_left=(3.3, 3.0), bottom_right=(4.3, 4.0), label="a", score=0.9),
                    ],
                    FP=[
                        ScoredLabeledBoundingBox(top_left=(5.5, 5.0), bottom_right=(6.5, 6.0), label="a", score=0.9),
                        ScoredLabeledBoundingBox(top_left=(7.7, 7.0), bottom_right=(8.7, 8.0), label="a", score=0.9),
                    ],
                    FN=[
                        LabeledBoundingBox(top_left=(5.0, 5.0), bottom_right=(6.0, 6.0), label="a"),
                        LabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(8.0, 8.0), label="a"),
                    ],
                    count_TP=2,
                    count_FP=2,
                    count_FN=2,
                    has_TP=True,
                    has_FP=True,
                    has_FN=True,
                    max_confidence_above_t=0.9,
                    min_confidence_above_t=0.9,
                    thresholds=0.5,
                ),
            ),
        ],
        "multiple bboxes in an image, varied confidence": [
            (
                TestSample(locator=fake_locator(14, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[
                        ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="a", score=0.6),
                        ScoredLabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="a", score=0.5),
                    ],
                    FP=[],
                    FN=[
                        LabeledBoundingBox(top_left=(5.0, 5.0), bottom_right=(6.0, 6.0), label="a"),
                        LabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(8.0, 8.0), label="a"),
                    ],
                    count_TP=2,
                    count_FP=0,
                    count_FN=2,
                    has_TP=True,
                    has_FP=False,
                    has_FN=True,
                    max_confidence_above_t=0.6,
                    min_confidence_above_t=0.5,
                    thresholds=0.5,
                ),
            ),
        ],
        "multiple bboxes in an image, many inferences": [
            (
                TestSample(locator=fake_locator(15, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[
                        ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(5.0, 5.0), bottom_right=(6.0, 6.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(8.0, 8.0), label="a", score=0.99),
                    ],
                    FP=[
                        ScoredLabeledBoundingBox(top_left=(0.0, 0.0), bottom_right=(1.0, 1.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(0.0, 0.0), bottom_right=(1.0, 1.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(0.0, 0.0), bottom_right=(1.0, 1.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(0.0, 0.0), bottom_right=(1.0, 1.0), label="a", score=0.99),
                    ],
                    FN=[],
                    count_TP=4,
                    count_FP=4,
                    count_FN=0,
                    has_TP=True,
                    has_FP=True,
                    has_FN=False,
                    max_confidence_above_t=0.99,
                    min_confidence_above_t=0.99,
                    thresholds=0.5,
                ),
            ),
        ],
        "multiple bboxes in an image, too few inferences": [
            (
                TestSample(locator=fake_locator(16, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[
                        ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(8.0, 8.0), label="a", score=0.99),
                    ],
                    FP=[],
                    FN=[
                        LabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="a"),
                        LabeledBoundingBox(top_left=(5.0, 5.0), bottom_right=(6.0, 6.0), label="a"),
                    ],
                    count_TP=2,
                    count_FP=0,
                    count_FN=2,
                    has_TP=True,
                    has_FP=False,
                    has_FN=True,
                    max_confidence_above_t=0.99,
                    min_confidence_above_t=0.99,
                    thresholds=0.5,
                ),
            ),
        ],
        "multiple bboxes in an image, suboptimal infs": [
            (
                TestSample(locator=fake_locator(17, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[ScoredLabeledBoundingBox(top_left=(5.0, 5.0), bottom_right=(6.0, 6.0), label="a", score=0.9)],
                    FP=[ScoredLabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(9.0, 9.0), label="a", score=0.9)],
                    FN=[
                        LabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="a"),
                        LabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(8.0, 8.0), label="a"),
                        LabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="a"),
                    ],
                    count_TP=1,
                    count_FP=1,
                    count_FN=3,
                    has_TP=True,
                    has_FP=True,
                    has_FN=True,
                    max_confidence_above_t=0.9,
                    min_confidence_above_t=0.9,
                    thresholds=0.5,
                ),
            ),
        ],
        "multiple bboxes in an image, ignored matches": [
            (
                TestSample(locator=fake_locator(18, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[
                        ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="a", score=0.9),
                        ScoredLabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="a", score=0.9),
                    ],
                    FP=[],
                    FN=[],
                    count_TP=2,
                    count_FP=0,
                    count_FN=0,
                    has_TP=True,
                    has_FP=False,
                    has_FN=False,
                    max_confidence_above_t=0.9,
                    min_confidence_above_t=0.9,
                    thresholds=0.5,
                ),
            ),
        ],
    },
    "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.3": {
        "nothing": [
            (
                TestSample(locator=fake_locator(1, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[],
                    FN=[],
                    count_TP=0,
                    count_FP=0,
                    count_FN=0,
                    has_TP=False,
                    has_FP=False,
                    has_FN=False,
                    max_confidence_above_t=None,
                    min_confidence_above_t=None,
                    thresholds=0.5,
                ),
            ),
        ],
        "no inferences": [
            (
                TestSample(locator=fake_locator(2, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[],
                    FN=[LabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="b")],
                    count_TP=0,
                    count_FP=0,
                    count_FN=1,
                    has_TP=False,
                    has_FP=False,
                    has_FN=True,
                    max_confidence_above_t=None,
                    min_confidence_above_t=None,
                    thresholds=0.5,
                ),
            ),
        ],
        "no ground truths": [
            (
                TestSample(locator=fake_locator(3, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="b", score=1.0)],
                    FN=[],
                    count_TP=0,
                    count_FP=1,
                    count_FN=0,
                    has_TP=False,
                    has_FP=True,
                    has_FN=False,
                    max_confidence_above_t=1.0,
                    min_confidence_above_t=1.0,
                    thresholds=0.5,
                ),
            ),
        ],
        "iou=1 and different labels and max confidence": [
            (
                TestSample(locator=fake_locator(4, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="b", score=1.0)],
                    FP=[],
                    FN=[],
                    count_TP=1,
                    count_FP=0,
                    count_FN=0,
                    has_TP=True,
                    has_FP=False,
                    has_FN=False,
                    max_confidence_above_t=1.0,
                    min_confidence_above_t=1.0,
                    thresholds=0.5,
                ),
            ),
        ],
        "iou=0 and same labels": [
            (
                TestSample(locator=fake_locator(5, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="b", score=1.0)],
                    FN=[LabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="b")],
                    count_TP=0,
                    count_FP=1,
                    count_FN=1,
                    has_TP=False,
                    has_FP=True,
                    has_FN=True,
                    max_confidence_above_t=1.0,
                    min_confidence_above_t=1.0,
                    thresholds=0.5,
                ),
            ),
        ],
        "iou=0.33 and same labels but 0 confidence": [
            (
                TestSample(locator=fake_locator(6, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[],
                    FN=[LabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(4.0, 4.0), label="b")],
                    count_TP=0,
                    count_FP=0,
                    count_FN=1,
                    has_TP=False,
                    has_FP=False,
                    has_FN=True,
                    max_confidence_above_t=None,
                    min_confidence_above_t=None,
                    thresholds=0.5,
                ),
            ),
        ],
        "iou=0.33 and same labels but 0.5 confidence": [
            (
                TestSample(locator=fake_locator(7, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[ScoredLabeledBoundingBox(top_left=(2.0, 2.0), bottom_right=(5.0, 5.0), label="b", score=0.5)],
                    FN=[LabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(4.0, 4.0), label="b")],
                    count_TP=0,
                    count_FP=1,
                    count_FN=1,
                    has_TP=False,
                    has_FP=True,
                    has_FN=True,
                    max_confidence_above_t=0.5,
                    min_confidence_above_t=0.5,
                    thresholds=0.5,
                ),
            ),
        ],
        "iou=0.33 and same labels but 0.99 confidence": [
            (
                TestSample(locator=fake_locator(8, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[ScoredLabeledBoundingBox(top_left=(2.0, 2.0), bottom_right=(5.0, 5.0), label="b", score=0.99)],
                    FN=[LabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(4.0, 4.0), label="b")],
                    count_TP=0,
                    count_FP=1,
                    count_FN=1,
                    has_TP=False,
                    has_FP=True,
                    has_FN=True,
                    max_confidence_above_t=0.99,
                    min_confidence_above_t=0.99,
                    thresholds=0.5,
                ),
            ),
        ],
        "iou=0.5 and same labels but 0 confidence": [
            (
                TestSample(locator=fake_locator(9, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[],
                    FN=[LabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(4.0, 4.0), label="b")],
                    count_TP=0,
                    count_FP=0,
                    count_FN=1,
                    has_TP=False,
                    has_FP=False,
                    has_FN=True,
                    max_confidence_above_t=None,
                    min_confidence_above_t=None,
                    thresholds=0.5,
                ),
            ),
        ],
        "iou=0.5 and same labels but 0.5 confidence": [
            (
                TestSample(locator=fake_locator(10, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[ScoredLabeledBoundingBox(top_left=(2.0, 1.0), bottom_right=(5.0, 4.0), label="b", score=0.5)],
                    FP=[],
                    FN=[],
                    count_TP=1,
                    count_FP=0,
                    count_FN=0,
                    has_TP=True,
                    has_FP=False,
                    has_FN=False,
                    max_confidence_above_t=0.5,
                    min_confidence_above_t=0.5,
                    thresholds=0.5,
                ),
            ),
        ],
        "iou=0.5 and same labels but 0.99 confidence": [
            (
                TestSample(locator=fake_locator(11, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[ScoredLabeledBoundingBox(top_left=(2.0, 1.0), bottom_right=(5.0, 4.0), label="b", score=0.99)],
                    FP=[],
                    FN=[],
                    count_TP=1,
                    count_FP=0,
                    count_FN=0,
                    has_TP=True,
                    has_FP=False,
                    has_FN=False,
                    max_confidence_above_t=0.99,
                    min_confidence_above_t=0.99,
                    thresholds=0.5,
                ),
            ),
        ],
        "multiple bboxes in an image, perfect match": [
            (
                TestSample(locator=fake_locator(12, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[
                        ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="a", score=0.9),
                        ScoredLabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="a", score=0.9),
                        ScoredLabeledBoundingBox(top_left=(5.0, 5.0), bottom_right=(6.0, 6.0), label="a", score=0.9),
                        ScoredLabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(8.0, 8.0), label="a", score=0.9),
                    ],
                    FP=[],
                    FN=[],
                    count_TP=4,
                    count_FP=0,
                    count_FN=0,
                    has_TP=True,
                    has_FP=False,
                    has_FN=False,
                    max_confidence_above_t=0.9,
                    min_confidence_above_t=0.9,
                    thresholds=0.5,
                ),
            ),
        ],
        "multiple bboxes in an image, varied iou": [
            (
                TestSample(locator=fake_locator(13, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[
                        ScoredLabeledBoundingBox(top_left=(1.1, 1.0), bottom_right=(2.1, 2.0), label="a", score=0.9),
                        ScoredLabeledBoundingBox(top_left=(3.3, 3.0), bottom_right=(4.3, 4.0), label="a", score=0.9),
                    ],
                    FP=[
                        ScoredLabeledBoundingBox(top_left=(5.5, 5.0), bottom_right=(6.5, 6.0), label="a", score=0.9),
                        ScoredLabeledBoundingBox(top_left=(7.7, 7.0), bottom_right=(8.7, 8.0), label="a", score=0.9),
                    ],
                    FN=[
                        LabeledBoundingBox(top_left=(5.0, 5.0), bottom_right=(6.0, 6.0), label="a"),
                        LabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(8.0, 8.0), label="a"),
                    ],
                    count_TP=2,
                    count_FP=2,
                    count_FN=2,
                    has_TP=True,
                    has_FP=True,
                    has_FN=True,
                    max_confidence_above_t=0.9,
                    min_confidence_above_t=0.9,
                    thresholds=0.5,
                ),
            ),
        ],
        "multiple bboxes in an image, varied confidence": [
            (
                TestSample(locator=fake_locator(14, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[
                        ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="a", score=0.6),
                        ScoredLabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="a", score=0.5),
                    ],
                    FP=[],
                    FN=[
                        LabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(8.0, 8.0), label="a"),
                        LabeledBoundingBox(top_left=(5.0, 5.0), bottom_right=(6.0, 6.0), label="a"),
                    ],
                    count_TP=2,
                    count_FP=0,
                    count_FN=2,
                    has_TP=True,
                    has_FP=False,
                    has_FN=True,
                    max_confidence_above_t=0.6,
                    min_confidence_above_t=0.5,
                    thresholds=0.5,
                ),
            ),
        ],
        "multiple bboxes in an image, many inferences": [
            (
                TestSample(locator=fake_locator(15, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[
                        ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(5.0, 5.0), bottom_right=(6.0, 6.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(8.0, 8.0), label="a", score=0.99),
                    ],
                    FP=[
                        ScoredLabeledBoundingBox(top_left=(0.0, 0.0), bottom_right=(1.0, 1.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(0.0, 0.0), bottom_right=(1.0, 1.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(0.0, 0.0), bottom_right=(1.0, 1.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(0.0, 0.0), bottom_right=(1.0, 1.0), label="a", score=0.99),
                    ],
                    FN=[],
                    count_TP=4,
                    count_FP=4,
                    count_FN=0,
                    has_TP=True,
                    has_FP=True,
                    has_FN=False,
                    max_confidence_above_t=0.99,
                    min_confidence_above_t=0.99,
                    thresholds=0.5,
                ),
            ),
        ],
        "multiple bboxes in an image, too few inferences": [
            (
                TestSample(locator=fake_locator(16, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[
                        ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(8.0, 8.0), label="a", score=0.99),
                    ],
                    FP=[],
                    FN=[
                        LabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="a"),
                        LabeledBoundingBox(top_left=(5.0, 5.0), bottom_right=(6.0, 6.0), label="a"),
                    ],
                    count_TP=2,
                    count_FP=0,
                    count_FN=2,
                    has_TP=True,
                    has_FP=False,
                    has_FN=True,
                    max_confidence_above_t=0.99,
                    min_confidence_above_t=0.99,
                    thresholds=0.5,
                ),
            ),
        ],
        "multiple bboxes in an image, suboptimal infs": [
            (
                TestSample(locator=fake_locator(17, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[ScoredLabeledBoundingBox(top_left=(5.0, 5.0), bottom_right=(6.0, 6.0), label="a", score=0.9)],
                    FP=[ScoredLabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(9.0, 9.0), label="a", score=0.9)],
                    FN=[
                        LabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="a"),
                        LabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="a"),
                        LabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(8.0, 8.0), label="a"),
                    ],
                    count_TP=1,
                    count_FP=1,
                    count_FN=3,
                    has_TP=True,
                    has_FP=True,
                    has_FN=True,
                    max_confidence_above_t=0.9,
                    min_confidence_above_t=0.9,
                    thresholds=0.5,
                ),
            ),
        ],
        "multiple bboxes in an image, ignored matches": [
            (
                TestSample(locator=fake_locator(18, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[
                        ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="a", score=0.9),
                        ScoredLabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="a", score=0.9),
                    ],
                    FP=[],
                    FN=[],
                    count_TP=2,
                    count_FP=0,
                    count_FN=0,
                    has_TP=True,
                    has_FP=False,
                    has_FN=False,
                    max_confidence_above_t=0.9,
                    min_confidence_above_t=0.9,
                    thresholds=0.5,
                ),
            ),
        ],
    },
    "Threshold: F1-Optimal, IoU: 0.5, confidence ≥ 0.1": {
        "nothing": [
            (
                TestSample(locator=fake_locator(1, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[],
                    FN=[],
                    count_TP=0,
                    count_FP=0,
                    count_FN=0,
                    has_TP=False,
                    has_FP=False,
                    has_FN=False,
                    max_confidence_above_t=None,
                    min_confidence_above_t=None,
                    thresholds=0.1,
                ),
            ),
        ],
        "no inferences": [
            (
                TestSample(locator=fake_locator(2, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[],
                    FN=[LabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="b")],
                    count_TP=0,
                    count_FP=0,
                    count_FN=1,
                    has_TP=False,
                    has_FP=False,
                    has_FN=True,
                    max_confidence_above_t=None,
                    min_confidence_above_t=None,
                    thresholds=0.1,
                ),
            ),
        ],
        "no ground truths": [
            (
                TestSample(locator=fake_locator(3, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="b", score=1.0)],
                    FN=[],
                    count_TP=0,
                    count_FP=1,
                    count_FN=0,
                    has_TP=False,
                    has_FP=True,
                    has_FN=False,
                    max_confidence_above_t=1.0,
                    min_confidence_above_t=1.0,
                    thresholds=0.1,
                ),
            ),
        ],
        "iou=1 and different labels and max confidence": [
            (
                TestSample(locator=fake_locator(4, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="b", score=1.0)],
                    FP=[],
                    FN=[],
                    count_TP=1,
                    count_FP=0,
                    count_FN=0,
                    has_TP=True,
                    has_FP=False,
                    has_FN=False,
                    max_confidence_above_t=1.0,
                    min_confidence_above_t=1.0,
                    thresholds=0.1,
                ),
            ),
        ],
        "iou=0 and same labels": [
            (
                TestSample(locator=fake_locator(5, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="b", score=1.0)],
                    FN=[LabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="b")],
                    count_TP=0,
                    count_FP=1,
                    count_FN=1,
                    has_TP=False,
                    has_FP=True,
                    has_FN=True,
                    max_confidence_above_t=1.0,
                    min_confidence_above_t=1.0,
                    thresholds=0.1,
                ),
            ),
        ],
        "iou=0.33 and same labels but 0 confidence": [
            (
                TestSample(locator=fake_locator(6, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[],
                    FN=[LabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(4.0, 4.0), label="b")],
                    count_TP=0,
                    count_FP=0,
                    count_FN=1,
                    has_TP=False,
                    has_FP=False,
                    has_FN=True,
                    max_confidence_above_t=None,
                    min_confidence_above_t=None,
                    thresholds=0.1,
                ),
            ),
        ],
        "iou=0.33 and same labels but 0.5 confidence": [
            (
                TestSample(locator=fake_locator(7, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[ScoredLabeledBoundingBox(top_left=(2.0, 2.0), bottom_right=(5.0, 5.0), label="b", score=0.5)],
                    FN=[LabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(4.0, 4.0), label="b")],
                    count_TP=0,
                    count_FP=1,
                    count_FN=1,
                    has_TP=False,
                    has_FP=True,
                    has_FN=True,
                    max_confidence_above_t=0.5,
                    min_confidence_above_t=0.5,
                    thresholds=0.1,
                ),
            ),
        ],
        "iou=0.33 and same labels but 0.99 confidence": [
            (
                TestSample(locator=fake_locator(8, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[ScoredLabeledBoundingBox(top_left=(2.0, 2.0), bottom_right=(5.0, 5.0), label="b", score=0.99)],
                    FN=[LabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(4.0, 4.0), label="b")],
                    count_TP=0,
                    count_FP=1,
                    count_FN=1,
                    has_TP=False,
                    has_FP=True,
                    has_FN=True,
                    max_confidence_above_t=0.99,
                    min_confidence_above_t=0.99,
                    thresholds=0.1,
                ),
            ),
        ],
        "iou=0.5 and same labels but 0 confidence": [
            (
                TestSample(locator=fake_locator(9, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[],
                    FN=[LabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(4.0, 4.0), label="b")],
                    count_TP=0,
                    count_FP=0,
                    count_FN=1,
                    has_TP=False,
                    has_FP=False,
                    has_FN=True,
                    max_confidence_above_t=None,
                    min_confidence_above_t=None,
                    thresholds=0.1,
                ),
            ),
        ],
        "iou=0.5 and same labels but 0.5 confidence": [
            (
                TestSample(locator=fake_locator(10, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[ScoredLabeledBoundingBox(top_left=(2.0, 1.0), bottom_right=(5.0, 4.0), label="b", score=0.5)],
                    FP=[],
                    FN=[],
                    count_TP=1,
                    count_FP=0,
                    count_FN=0,
                    has_TP=True,
                    has_FP=False,
                    has_FN=False,
                    max_confidence_above_t=0.5,
                    min_confidence_above_t=0.5,
                    thresholds=0.1,
                ),
            ),
        ],
        "iou=0.5 and same labels but 0.99 confidence": [
            (
                TestSample(locator=fake_locator(11, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[ScoredLabeledBoundingBox(top_left=(2.0, 1.0), bottom_right=(5.0, 4.0), label="b", score=0.99)],
                    FP=[],
                    FN=[],
                    count_TP=1,
                    count_FP=0,
                    count_FN=0,
                    has_TP=True,
                    has_FP=False,
                    has_FN=False,
                    max_confidence_above_t=0.99,
                    min_confidence_above_t=0.99,
                    thresholds=0.1,
                ),
            ),
        ],
        "multiple bboxes in an image, perfect match": [
            (
                TestSample(locator=fake_locator(12, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[
                        ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="a", score=0.9),
                        ScoredLabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="a", score=0.9),
                        ScoredLabeledBoundingBox(top_left=(5.0, 5.0), bottom_right=(6.0, 6.0), label="a", score=0.9),
                        ScoredLabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(8.0, 8.0), label="a", score=0.9),
                    ],
                    FP=[],
                    FN=[],
                    count_TP=4,
                    count_FP=0,
                    count_FN=0,
                    has_TP=True,
                    has_FP=False,
                    has_FN=False,
                    max_confidence_above_t=0.9,
                    min_confidence_above_t=0.9,
                    thresholds=0.1,
                ),
            ),
        ],
        "multiple bboxes in an image, varied iou": [
            (
                TestSample(locator=fake_locator(13, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[
                        ScoredLabeledBoundingBox(top_left=(1.1, 1.0), bottom_right=(2.1, 2.0), label="a", score=0.9),
                        ScoredLabeledBoundingBox(top_left=(3.3, 3.0), bottom_right=(4.3, 4.0), label="a", score=0.9),
                    ],
                    FP=[
                        ScoredLabeledBoundingBox(top_left=(5.5, 5.0), bottom_right=(6.5, 6.0), label="a", score=0.9),
                        ScoredLabeledBoundingBox(top_left=(7.7, 7.0), bottom_right=(8.7, 8.0), label="a", score=0.9),
                    ],
                    FN=[
                        LabeledBoundingBox(top_left=(5.0, 5.0), bottom_right=(6.0, 6.0), label="a"),
                        LabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(8.0, 8.0), label="a"),
                    ],
                    count_TP=2,
                    count_FP=2,
                    count_FN=2,
                    has_TP=True,
                    has_FP=True,
                    has_FN=True,
                    max_confidence_above_t=0.9,
                    min_confidence_above_t=0.9,
                    thresholds=0.1,
                ),
            ),
        ],
        "multiple bboxes in an image, varied confidence": [
            (
                TestSample(locator=fake_locator(14, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[
                        ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="a", score=0.6),
                        ScoredLabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="a", score=0.5),
                        ScoredLabeledBoundingBox(top_left=(5.0, 5.0), bottom_right=(6.0, 6.0), label="a", score=0.4),
                    ],
                    FP=[],
                    FN=[LabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(8.0, 8.0), label="a")],
                    count_TP=3,
                    count_FP=0,
                    count_FN=1,
                    has_TP=True,
                    has_FP=False,
                    has_FN=True,
                    max_confidence_above_t=0.6,
                    min_confidence_above_t=0.4,
                    thresholds=0.1,
                ),
            ),
        ],
        "multiple bboxes in an image, many inferences": [
            (
                TestSample(locator=fake_locator(15, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[
                        ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(5.0, 5.0), bottom_right=(6.0, 6.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(8.0, 8.0), label="a", score=0.99),
                    ],
                    FP=[
                        ScoredLabeledBoundingBox(top_left=(0.0, 0.0), bottom_right=(1.0, 1.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(0.0, 0.0), bottom_right=(1.0, 1.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(0.0, 0.0), bottom_right=(1.0, 1.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(0.0, 0.0), bottom_right=(1.0, 1.0), label="a", score=0.99),
                    ],
                    FN=[],
                    count_TP=4,
                    count_FP=4,
                    count_FN=0,
                    has_TP=True,
                    has_FP=True,
                    has_FN=False,
                    max_confidence_above_t=0.99,
                    min_confidence_above_t=0.99,
                    thresholds=0.1,
                ),
            ),
        ],
        "multiple bboxes in an image, too few inferences": [
            (
                TestSample(locator=fake_locator(16, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[
                        ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="a", score=0.99),
                        ScoredLabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(8.0, 8.0), label="a", score=0.99),
                    ],
                    FP=[],
                    FN=[
                        LabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="a"),
                        LabeledBoundingBox(top_left=(5.0, 5.0), bottom_right=(6.0, 6.0), label="a"),
                    ],
                    count_TP=2,
                    count_FP=0,
                    count_FN=2,
                    has_TP=True,
                    has_FP=False,
                    has_FN=True,
                    max_confidence_above_t=0.99,
                    min_confidence_above_t=0.99,
                    thresholds=0.1,
                ),
            ),
        ],
        "multiple bboxes in an image, suboptimal infs": [
            (
                TestSample(locator=fake_locator(17, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[ScoredLabeledBoundingBox(top_left=(5.0, 5.0), bottom_right=(6.0, 6.0), label="a", score=0.9)],
                    FP=[ScoredLabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(9.0, 9.0), label="a", score=0.9)],
                    FN=[
                        LabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="a"),
                        LabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="a"),
                        LabeledBoundingBox(top_left=(7.0, 7.0), bottom_right=(8.0, 8.0), label="a"),
                    ],
                    count_TP=1,
                    count_FP=1,
                    count_FN=3,
                    has_TP=True,
                    has_FP=True,
                    has_FN=True,
                    max_confidence_above_t=0.9,
                    min_confidence_above_t=0.9,
                    thresholds=0.1,
                ),
            ),
        ],
        "multiple bboxes in an image, ignored matches": [
            (
                TestSample(locator=fake_locator(18, "OD"), metadata={}),
                TestSampleMetricsSingleClass(
                    TP=[
                        ScoredLabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(2.0, 2.0), label="a", score=0.9),
                        ScoredLabeledBoundingBox(top_left=(3.0, 3.0), bottom_right=(4.0, 4.0), label="a", score=0.9),
                    ],
                    FP=[],
                    FN=[],
                    count_TP=2,
                    count_FP=0,
                    count_FN=0,
                    has_TP=True,
                    has_FP=False,
                    has_FN=False,
                    max_confidence_above_t=0.9,
                    min_confidence_above_t=0.9,
                    thresholds=0.1,
                ),
            ),
        ],
    },
}


@pytest.mark.parametrize(
    "config_name, test_name",
    TEST_PARAMS,
)
def test__object_detection__single_class__compute__test_sample_metrics(
    config_name: str,
    test_name: str,
) -> None:
    config = TEST_CONFIGURATIONS[config_name]
    eval = ObjectDetectionEvaluator(configurations=[config])
    result = eval.compute_test_sample_metrics(
        test_case=TEST_CASE,
        inferences=TEST_DATA[test_name],
        configuration=config,
    )

    assert EXPECTED_COMPUTE_TEST_SAMPLE_METRICS[config_name][test_name] == result


def test__object_detection__single_class__compute__test_sample_metrics__all() -> None:
    for config_name, config in TEST_CONFIGURATIONS.items():
        if config_name not in EXPECTED_COMPUTE_TEST_SAMPLE_METRICS:
            continue
        eval = ObjectDetectionEvaluator(configurations=[config])
        result = eval.compute_test_sample_metrics(
            test_case=TEST_CASE,
            inferences=[ts_gt_inf for _, data in TEST_DATA.items() for ts_gt_inf in data],
            configuration=config,
        )

        assert [
            pair for _, results in EXPECTED_COMPUTE_TEST_SAMPLE_METRICS[config_name].items() for pair in results
        ] == result
