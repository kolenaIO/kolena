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

from kolena.workflow.annotation import LabeledPolygon, ScoredLabel
from kolena.workflow.annotation import ScoredLabeledPolygon
from kolena.workflow.plot import ConfusionMatrix
from kolena.workflow.plot import Curve
from kolena.workflow.plot import CurvePlot
from tests.integration.helper import fake_locator
from tests.integration.helper import with_test_prefix


instance_segmentation = pytest.importorskip(
    "kolena._experimental.instance_segmentation",
    reason="requires kolena[metrics] extra",
)
GroundTruth = instance_segmentation.GroundTruth
Inference = instance_segmentation.Inference
TestCase = instance_segmentation.TestCase
TestSample = instance_segmentation.TestSample
TestSuite = instance_segmentation.TestSuite
InstanceSegmentationEvaluator = instance_segmentation.InstanceSegmentationEvaluator
ThresholdConfiguration = instance_segmentation.ThresholdConfiguration
ClassMetricsPerTestCase = instance_segmentation.ClassMetricsPerTestCase
TestCaseMetrics = instance_segmentation.TestCaseMetrics
TestSampleMetrics = instance_segmentation.TestSampleMetrics
TestSuiteMetrics = instance_segmentation.TestSuiteMetrics


TEST_DATA: List[Tuple[TestSample, GroundTruth, Inference]] = [
    (
        TestSample(locator=fake_locator(112, "IS")),
        GroundTruth(
            polygons=[
                LabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a"),
                LabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "b"),
                LabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "c"),
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "d"),
            ],
        ),
        Inference(
            polygons=[
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 1),
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "b", 0.9),
                ScoredLabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "c", 0.8),
                ScoredLabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "d", 0.7),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(113, "IS")),
        GroundTruth(
            polygons=[
                LabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a"),
                LabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a"),
                LabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a"),
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "b"),
            ],
        ),
        Inference(
            polygons=[
                ScoredLabeledPolygon([(1.1, 1), (1.1, 2), (2, 1.1), (2, 2)], "a", 0.9),
                ScoredLabeledPolygon([(3.2, 3), (3.2, 4), (4, 3.2), (4, 4)], "a", 0.8),
                ScoredLabeledPolygon([(5.3, 5), (5.3, 6), (6, 5.3), (6, 6)], "a", 0.7),
                ScoredLabeledPolygon([(7.4, 7), (7.4, 8), (8, 7.4), (8, 8)], "b", 1),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(114, "IS")),
        GroundTruth(
            polygons=[
                LabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a"),
                LabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "b"),
                LabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "b"),
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a"),
            ],
        ),
        Inference(
            polygons=[
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 0.6),
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "b", 0.5),
                ScoredLabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "b", 0.4),
                ScoredLabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a", 0.1),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(115, "IS")),
        GroundTruth(
            polygons=[
                LabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a"),
                LabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "b"),
                LabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "b"),
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a"),
            ],
        ),
        Inference(
            polygons=[
                ScoredLabeledPolygon([(0, 0), (0, 1), (1, 0), (1, 1)], "a", 1),
                ScoredLabeledPolygon([(0, 0), (0, 1), (1, 0), (1, 1)], "a", 1),
                ScoredLabeledPolygon([(0, 0), (0, 1), (1, 0), (1, 1)], "a", 0.9),
                ScoredLabeledPolygon([(0, 0), (0, 1), (1, 0), (1, 1)], "a", 0.8),
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 1),
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "b", 0.9),
                ScoredLabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "b", 0.9),
                ScoredLabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a", 0.8),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(116, "IS")),
        GroundTruth(
            polygons=[
                LabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a"),
                LabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "b"),
                LabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "b"),
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a"),
            ],
        ),
        Inference(
            polygons=[
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 1),
                ScoredLabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a", 0.4),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(117, "IS")),
        GroundTruth(
            polygons=[
                LabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a"),
                LabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "c"),
                LabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a"),
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a"),
            ],
        ),
        Inference(
            polygons=[
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 0),
                ScoredLabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a", 0.9),
                ScoredLabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "b", 0.9),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(118, "IS")),
        GroundTruth(
            polygons=[
                LabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a"),
                LabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "c"),
            ],
            ignored_polygons=[
                LabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "b"),
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a"),
            ],
        ),
        Inference(
            polygons=[
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 0.9),
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "c", 0.8),
                ScoredLabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "b", 0.1),
                ScoredLabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a", 1),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(119, "IS")),
        GroundTruth(
            polygons=[
                LabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "e"),
                LabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "e"),
                LabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "e"),
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "e"),
                LabeledPolygon([(9, 9), (9, 10), (10, 9), (10, 10)], "e"),
                LabeledPolygon([(11, 11), (11, 12), (12, 11), (12, 12)], "e"),
                LabeledPolygon([(13, 13), (13, 14), (14, 13), (14, 14)], "e"),
                LabeledPolygon([(15, 15), (15, 16), (16, 15), (16, 16)], "e"),
                LabeledPolygon([(17, 17), (17, 18), (18, 17), (18, 18)], "e"),
                LabeledPolygon([(19, 19), (19, 20), (20, 19), (20, 20)], "e"),
            ],
            ignored_polygons=[
                LabeledPolygon([(2, 2), (2, 3), (3, 2), (3, 3)], "b"),
                LabeledPolygon([(4, 4), (4, 5), (5, 4), (5, 5)], "b"),
                LabeledPolygon([(6, 6), (6, 7), (7, 6), (7, 7)], "b"),
                LabeledPolygon([(8, 8), (8, 9), (9, 8), (9, 9)], "b"),
                LabeledPolygon([(10, 10), (10, 11), (11, 10), (11, 11)], "a"),
                LabeledPolygon([(12, 12), (12, 13), (13, 12), (13, 13)], "a"),
                LabeledPolygon([(14, 14), (14, 15), (15, 14), (15, 15)], "a"),
                LabeledPolygon([(16, 16), (16, 17), (17, 16), (17, 17)], "a"),
            ],
        ),
        Inference(
            polygons=[
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "b", 0.9),
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "e", 0.8),
                ScoredLabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "e", 0.7),
                ScoredLabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "e", 0.6),
                ScoredLabeledPolygon([(9, 9), (9, 10), (10, 9), (10, 10)], "e", 0.1),
                ScoredLabeledPolygon([(11, 11), (11, 12), (12, 11), (12, 12)], "e", 0),
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 0.9),
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a", 0.8),
                ScoredLabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a", 0.1),
                ScoredLabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a", 1),
                ScoredLabeledPolygon([(6, 6), (6, 7), (7, 6), (7, 7)], "b", 0.9),
                ScoredLabeledPolygon([(8, 8), (8, 9), (9, 8), (9, 9)], "b", 0.8),
                ScoredLabeledPolygon([(10, 10), (10, 11), (11, 10), (11, 11)], "b", 0.7),
                ScoredLabeledPolygon([(12, 12), (12, 13), (13, 12), (13, 13)], "b", 0.6),
                ScoredLabeledPolygon([(14, 14), (14, 15), (15, 14), (15, 15)], "b", 0.1),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(120, "IS")),
        GroundTruth(
            polygons=[
                LabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a"),
                LabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "b"),
                LabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "c"),
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "d"),
            ],
        ),
        Inference(
            polygons=[
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 1),
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "b", 0.9),
                ScoredLabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "c", 0.8),
                ScoredLabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "d", 0.7),
            ],
            ignored=True,
        ),
    ),
]


EXPECTED_COMPUTE_TEST_SAMPLE_METRICS: List[Tuple[TestSample, TestSampleMetrics]] = [
    (
        TestSample(locator=fake_locator(112, "IS"), metadata={}),
        TestSampleMetrics(
            TP=[
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 1),
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "b", 0.9),
                ScoredLabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "c", 0.8),
                ScoredLabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "d", 0.7),
            ],
            FP=[],
            FN=[],
            Confused=[],
            count_TP=4,
            count_FP=0,
            count_FN=0,
            count_Confused=0,
            has_TP=True,
            has_FP=False,
            has_FN=False,
            has_Confused=False,
            ignored=False,
            max_confidence_above_t=1,
            min_confidence_above_t=0.7,
            thresholds=[
                ScoredLabel("a", 0.5),
                ScoredLabel("b", 0.5),
                ScoredLabel("c", 0.5),
                ScoredLabel("d", 0.5),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(113, "IS"), metadata={}),
        TestSampleMetrics(
            TP=[
                ScoredLabeledPolygon([(1.1, 1), (1.1, 2), (2, 1.1), (2, 2)], "a", 0.9),
            ],
            FP=[
                ScoredLabeledPolygon([(3.2, 3), (3.2, 4), (4, 3.2), (4, 4)], "a", 0.8),
                ScoredLabeledPolygon([(5.3, 5), (5.3, 6), (6, 5.3), (6, 6)], "a", 0.7),
                ScoredLabeledPolygon([(7.4, 7), (7.4, 8), (8, 7.4), (8, 8)], "b", 1),
            ],
            FN=[
                LabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a"),
                LabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a"),
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "b"),
            ],
            Confused=[],
            count_TP=1,
            count_FP=3,
            count_FN=3,
            count_Confused=0,
            has_TP=True,
            has_FP=True,
            has_FN=True,
            has_Confused=False,
            ignored=False,
            max_confidence_above_t=1,
            min_confidence_above_t=0.7,
            thresholds=[
                ScoredLabel("a", 0.5),
                ScoredLabel("b", 0.5),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(114, "IS"), metadata={}),
        TestSampleMetrics(
            TP=[
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 0.6),
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "b", 0.5),
            ],
            FP=[],
            FN=[
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a"),
                LabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "b"),
            ],
            Confused=[],
            count_TP=2,
            count_FP=0,
            count_FN=2,
            count_Confused=0,
            has_TP=True,
            has_FP=False,
            has_FN=True,
            has_Confused=False,
            ignored=False,
            max_confidence_above_t=0.6,
            min_confidence_above_t=0.5,
            thresholds=[
                ScoredLabel("a", 0.5),
                ScoredLabel("b", 0.5),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(115, "IS"), metadata={}),
        TestSampleMetrics(
            TP=[
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 1),
                ScoredLabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a", 0.8),
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "b", 0.9),
                ScoredLabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "b", 0.9),
            ],
            FP=[
                ScoredLabeledPolygon([(0, 0), (0, 1), (1, 0), (1, 1)], "a", 1),
                ScoredLabeledPolygon([(0, 0), (0, 1), (1, 0), (1, 1)], "a", 1),
                ScoredLabeledPolygon([(0, 0), (0, 1), (1, 0), (1, 1)], "a", 0.9),
                ScoredLabeledPolygon([(0, 0), (0, 1), (1, 0), (1, 1)], "a", 0.8),
            ],
            FN=[],
            Confused=[],
            count_TP=4,
            count_FP=4,
            count_FN=0,
            count_Confused=0,
            has_TP=True,
            has_FP=True,
            has_FN=False,
            has_Confused=False,
            ignored=False,
            max_confidence_above_t=1,
            min_confidence_above_t=0.8,
            thresholds=[
                ScoredLabel("a", 0.5),
                ScoredLabel("b", 0.5),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(116, "IS"), metadata={}),
        TestSampleMetrics(
            TP=[
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 1),
            ],
            FP=[],
            FN=[
                LabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "b"),
                LabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "b"),
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a"),
            ],
            Confused=[],
            count_TP=1,
            count_FP=0,
            count_FN=3,
            count_Confused=0,
            has_TP=True,
            has_FP=False,
            has_FN=True,
            has_Confused=False,
            ignored=False,
            max_confidence_above_t=1,
            min_confidence_above_t=1,
            thresholds=[
                ScoredLabel("a", 0.5),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(117, "IS"), metadata={}),
        TestSampleMetrics(
            TP=[ScoredLabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a", 0.9)],
            FP=[ScoredLabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "b", 0.9)],
            FN=[
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a"),
                LabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "c"),
                LabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a"),
            ],
            Confused=[ScoredLabeledPolygon([(7.0, 7.0), (7.0, 8.0), (8.0, 7.0), (8.0, 8.0)], 'b', 0.9)],
            count_TP=1,
            count_FP=1,
            count_FN=3,
            count_Confused=1,
            has_TP=True,
            has_FP=True,
            has_FN=True,
            has_Confused=True,
            ignored=False,
            max_confidence_above_t=0.9,
            min_confidence_above_t=0.9,
            thresholds=[
                ScoredLabel("a", 0.5),
                ScoredLabel("b", 0.5),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(118, "IS"), metadata={}),
        TestSampleMetrics(
            TP=[
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 0.9),
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "c", 0.8),
            ],
            FP=[],
            FN=[],
            Confused=[],
            count_TP=2,
            count_FP=0,
            count_FN=0,
            count_Confused=0,
            has_TP=True,
            has_FP=False,
            has_FN=False,
            has_Confused=False,
            ignored=False,
            max_confidence_above_t=0.9,
            min_confidence_above_t=0.8,
            thresholds=[
                ScoredLabel("a", 0.5),
                ScoredLabel("c", 0.5),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(119, "IS"), metadata={}),
        TestSampleMetrics(
            TP=[
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "e", 0.8),
                ScoredLabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "e", 0.7),
                ScoredLabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "e", 0.6),
            ],
            FP=[
                ScoredLabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a", 1),
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 0.9),
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a", 0.8),
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "b", 0.9),
                ScoredLabeledPolygon([(10, 10), (10, 11), (11, 10), (11, 11)], "b", 0.7),
                ScoredLabeledPolygon([(12, 12), (12, 13), (13, 12), (13, 13)], "b", 0.6),
            ],
            FN=[
                LabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "e"),
                LabeledPolygon([(13, 13), (13, 14), (14, 13), (14, 14)], "e"),
                LabeledPolygon([(15, 15), (15, 16), (16, 15), (16, 16)], "e"),
                LabeledPolygon([(17, 17), (17, 18), (18, 17), (18, 18)], "e"),
                LabeledPolygon([(19, 19), (19, 20), (20, 19), (20, 20)], "e"),
                LabeledPolygon([(9, 9), (9, 10), (10, 9), (10, 10)], "e"),
                LabeledPolygon([(11, 11), (11, 12), (12, 11), (12, 12)], "e"),
            ],
            Confused=[ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 0.9)],
            count_TP=3,
            count_FP=6,
            count_FN=7,
            count_Confused=1,
            has_TP=True,
            has_FP=True,
            has_FN=True,
            has_Confused=True,
            ignored=False,
            max_confidence_above_t=1,
            min_confidence_above_t=0.6,
            thresholds=[
                ScoredLabel("a", 0.5),
                ScoredLabel("b", 0.5),
                ScoredLabel("e", 0.5),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(120, "IS"), metadata={}),
        TestSampleMetrics(
            TP=[],
            FP=[],
            FN=[],
            Confused=[],
            count_TP=0,
            count_FP=0,
            count_FN=0,
            count_Confused=0,
            has_TP=False,
            has_FP=False,
            has_FN=False,
            has_Confused=False,
            ignored=True,
            max_confidence_above_t=None,
            min_confidence_above_t=None,
            thresholds=[],
        ),
    ),

]


EXPECTED_COMPUTE_TEST_CASE_METRICS = TestCaseMetrics(
    PerClass=[
        ClassMetricsPerTestCase(
            Class="a",
            nImages=7,
            Threshold=0.5,
            Objects=14,
            Inferences=17,
            TP=8,
            FN=6,
            FP=9,
            Precision=8 / 17,
            Recall=4 / 7,
            F1=16 / 31,
            AP=210 / 499,
        ),
        ClassMetricsPerTestCase(
            Class="b",
            nImages=5,
            Threshold=0.5,
            Objects=8,
            Inferences=9,
            TP=4,
            FN=4,
            FP=5,
            Precision=4 / 9,
            Recall=0.5,
            F1=8 / 17,
            AP=5 / 16,
        ),
        ClassMetricsPerTestCase(
            Class="c",
            nImages=3,
            Threshold=0.5,
            Objects=3,
            Inferences=2,
            TP=2,
            FN=1,
            FP=0,
            Precision=1,
            Recall=2 / 3,
            F1=0.8,
            AP=2 / 3,
        ),
        ClassMetricsPerTestCase(
            Class="d",
            nImages=1,
            Threshold=0.5,
            Objects=1,
            Inferences=1,
            TP=1,
            FN=0,
            FP=0,
            Precision=1,
            Recall=1,
            F1=1,
            AP=1,
        ),
        ClassMetricsPerTestCase(
            Class="e",
            nImages=1,
            Threshold=0.5,
            Objects=10,
            Inferences=3,
            TP=3,
            FN=7,
            FP=0,
            Precision=1,
            Recall=0.3,
            F1=6 / 13,
            AP=0.5,
        ),
    ],
    Objects=36,
    Inferences=32,
    TP=18,
    FN=18,
    FP=14,
    nIgnored=1,
    macro_Precision=599 / 765,
    macro_Recall=319 / 525,
    macro_F1=280 / 431,
    mean_AP=29 / 50,
    micro_Precision=9 / 16,
    micro_Recall=0.5,
    micro_F1=9 / 17,
)


EXPECTED_CONFUSION_MATRIX = ConfusionMatrix(
    title="Confusion Matrix",
    labels=["a", "b", "c", "d", "e"],
    matrix=[[8, 1, 0, 0, 0], [0, 4, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 1, 0], [1, 0, 0, 0, 3]],
    x_label="Predicted",
    y_label="Actual",
)


EXPECTED_F1_CURVE_PLOT = CurvePlot(
    title="F1-Score vs. Confidence Threshold Per Class",
    x_label="Confidence Threshold",
    y_label="F1-Score",
    curves=[
        Curve(
            x=[0, 0.1, 0.4, 0.6, 0.8, 0.9, 1],
            y=[22 / 35, 10 / 17, 9 / 16, 16 / 31, 14 / 29, 12 / 25, 0.3],
            label="a",
            extra={
                "Precision": [11 / 21, 0.5, 0.5, 8 / 17, 7 / 15, 6 / 11, 0.5],
                "Recall": [11 / 14, 5 / 7, 9 / 14, 4 / 7, 0.5, 3 / 7, 3 / 14],
            },
        ),
        Curve(
            x=[0.4, 0.5, 0.9, 1],
            y=[5 / 9, 8 / 17, 3 / 7, 0],
            label="b",
            extra={
                "Precision": [0.5, 4 / 9, 0.5, 0],
                "Recall": [5 / 8, 0.5, 3 / 8, 0],
            },
        ),
        Curve(
            x=[0, 0.1, 0.6, 0.7, 0.8],
            y=[2 / 3, 4 / 7, 6 / 13, 1 / 3, 2 / 11],
            label="e",
            extra={
                "Precision": [1.0, 1.0, 1.0, 1.0, 1.0],
                "Recall": [0.5, 0.4, 0.3, 0.2, 0.1],
            },
        ),
    ],
    x_config=None,
    y_config=None,
)


def assert_test_case_metrics_equals_expected(
    metrics: TestCaseMetrics,
    other_metrics: TestCaseMetrics,
) -> None:
    assert len(metrics.PerClass) == len(other_metrics.PerClass)
    pytest_precision = 1e-5
    for pc_metric, expected_pc_metric in zip(metrics.PerClass, other_metrics.PerClass):
        assert pc_metric.Class == expected_pc_metric.Class
        assert pc_metric.nImages == expected_pc_metric.nImages
        assert pc_metric.Threshold == expected_pc_metric.Threshold
        assert pc_metric.Objects == expected_pc_metric.Objects
        assert pc_metric.Inferences == expected_pc_metric.Inferences
        assert pc_metric.TP == expected_pc_metric.TP
        assert pc_metric.FN == expected_pc_metric.FN
        assert pc_metric.FP == expected_pc_metric.FP
        assert pytest.approx(pc_metric.Precision, abs=pytest_precision) == expected_pc_metric.Precision
        assert pytest.approx(pc_metric.Recall, abs=pytest_precision) == expected_pc_metric.Recall
        assert pytest.approx(pc_metric.F1, abs=pytest_precision) == expected_pc_metric.F1
        assert pytest.approx(pc_metric.AP, abs=pytest_precision) == expected_pc_metric.AP

    assert metrics.Objects == other_metrics.Objects
    assert metrics.Inferences == other_metrics.Inferences
    assert metrics.TP == other_metrics.TP
    assert metrics.FN == other_metrics.FN
    assert metrics.FP == other_metrics.FP
    assert pytest.approx(metrics.macro_Precision, abs=pytest_precision) == other_metrics.macro_Precision
    assert pytest.approx(metrics.macro_Recall, abs=pytest_precision) == other_metrics.macro_Recall
    assert pytest.approx(metrics.macro_F1, abs=pytest_precision) == other_metrics.macro_F1
    assert pytest.approx(metrics.mean_AP, abs=pytest_precision) == other_metrics.mean_AP


def assert_curves(
    curves: List[Curve],
    expected: List[Curve],
) -> None:
    assert len(curves) == len(expected)
    for curve, expectation in zip(curves, expected):
        print(curve, expectation)
        assert curve.label == expectation.label
        assert len(curve.x) == len(expectation.x)
        assert sum(abs(a - b) for a, b in zip(curve.x, expectation.x)) < 1e-12
        assert len(curve.y) == len(expectation.y)
        assert sum(abs(a - b) for a, b in zip(curve.y, expectation.y)) < 1e-12
        for extra_key in curve.extra.keys():
            assert sum(abs(a - b) for a, b in zip(curve.extra[extra_key], expectation.extra[extra_key])) < 1e-12


def assert_curve_plot_equal(
    plot: CurvePlot,
    expected: CurvePlot,
) -> None:
    assert plot.title == expected.title
    assert plot.x_label == expected.x_label
    assert plot.y_label == expected.y_label
    assert_curves(plot.curves, expected.curves)
    assert plot.x_config == expected.x_config
    assert plot.y_config == expected.y_config


@pytest.mark.metrics
def test__instance_segmentation__multiclass_evaluator__fixed() -> None:
    TEST_CASE_NAME = "multiclass IS test fixed"
    TEST_CASE = TestCase(with_test_prefix(TEST_CASE_NAME + " case"))
    TEST_SUITE = TestSuite(with_test_prefix(TEST_CASE_NAME + " suite"))
    config = ThresholdConfiguration(
        threshold_strategy=0.5,
        min_confidence_score=0,
    )

    eval = InstanceSegmentationEvaluator(configurations=[config])

    test_sample_metrics = eval.compute_test_sample_metrics(
        test_case=TEST_CASE,
        inferences=TEST_DATA,
        configuration=config,
    )

    assert config.display_name() not in eval.evaluator.threshold_cache
    assert len(eval.evaluator.matchings_by_test_case) != 0
    assert len(eval.evaluator.matchings_by_test_case[config.display_name()]) != 0
    num_of_ignored = sum([1 for _, _, inf in TEST_DATA if inf.ignored])
    assert (
        len(eval.evaluator.matchings_by_test_case[config.display_name()][TEST_CASE.name])
        == len(TEST_DATA) - num_of_ignored
    )
    assert test_sample_metrics == EXPECTED_COMPUTE_TEST_SAMPLE_METRICS

    test_case_metrics = eval.compute_test_case_metrics(
        test_case=TEST_CASE,
        inferences=TEST_DATA,
        metrics=[pair[1] for pair in EXPECTED_COMPUTE_TEST_SAMPLE_METRICS],
        configuration=config,
    )
    assert TEST_CASE.name in eval.evaluator.locators_by_test_case
    assert len(eval.evaluator.locators_by_test_case[TEST_CASE.name]) == len(TEST_DATA)
    assert_test_case_metrics_equals_expected(test_case_metrics, EXPECTED_COMPUTE_TEST_CASE_METRICS)

    # test case plots only use the cached values
    plots = eval.compute_test_case_plots(
        test_case=TEST_CASE,
        inferences=[],
        metrics=[],
        configuration=config,
    )
    assert_curve_plot_equal(plots[0], EXPECTED_F1_CURVE_PLOT)
    assert plots[2] == EXPECTED_CONFUSION_MATRIX

    # test suite metrics - one
    test_suite_metrics = eval.compute_test_suite_metrics(
        test_suite=TEST_SUITE,
        metrics=[
            (TEST_CASE, EXPECTED_COMPUTE_TEST_CASE_METRICS),
        ],
        configuration=config,
    )
    assert test_suite_metrics == TestSuiteMetrics(n_images=len(TEST_DATA), mean_AP=29 / 50)

    # test suite metrics - two
    test_suite_metrics_dup = eval.compute_test_suite_metrics(
        test_suite=TEST_SUITE,
        metrics=[
            (TEST_CASE, EXPECTED_COMPUTE_TEST_CASE_METRICS),
            (TEST_CASE, EXPECTED_COMPUTE_TEST_CASE_METRICS),
        ],
        configuration=config,
    )
    assert test_suite_metrics_dup == TestSuiteMetrics(n_images=len(TEST_DATA), mean_AP=29 / 50)
