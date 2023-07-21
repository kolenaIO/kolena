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

from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import ScoredClassificationLabel
from kolena.workflow.annotation import ScoredLabeledBoundingBox
from kolena.workflow.plot import ConfusionMatrix
from kolena.workflow.plot import Curve
from kolena.workflow.plot import CurvePlot
from kolena.workflow.plot import Plot
from tests.integration.helper import fake_locator
from tests.integration.helper import with_test_prefix

object_detection = pytest.importorskip("kolena._experimental.object_detection", reason="requires kolena[metrics] extra")
ClassMetricsPerTestCase = object_detection.ClassMetricsPerTestCase
GroundTruth = object_detection.GroundTruth
Inference = object_detection.Inference
TestCase = object_detection.TestCase
TestCaseMetrics = object_detection.TestCaseMetrics
TestSample = object_detection.TestSample
TestSampleMetrics = object_detection.TestSampleMetrics
TestSuite = object_detection.TestSuite
TestSuiteMetrics = object_detection.TestSuiteMetrics
ThresholdConfiguration = object_detection.ThresholdConfiguration
ThresholdStrategy = object_detection.ThresholdStrategy


TEST_CASE_NAME = "multiclass OD test"
TEST_CASE = TestCase(with_test_prefix(TEST_CASE_NAME + " case"))
TEST_SUITE = TestSuite(with_test_prefix(TEST_CASE_NAME + " suite"))


TEST_DATA: List[Tuple[TestSample, GroundTruth, Inference]] = [
    (
        TestSample(locator=fake_locator(112, "OD")),
        GroundTruth(
            bboxes=[
                LabeledBoundingBox((1, 1), (2, 2), "a"),
                LabeledBoundingBox((3, 3), (4, 4), "b"),
                LabeledBoundingBox((5, 5), (6, 6), "c"),
                LabeledBoundingBox((7, 7), (8, 8), "d"),
            ],
        ),
        Inference(
            bboxes=[
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 1),
                ScoredLabeledBoundingBox((3, 3), (4, 4), "b", 0.9),
                ScoredLabeledBoundingBox((5, 5), (6, 6), "c", 0.8),
                ScoredLabeledBoundingBox((7, 7), (8, 8), "d", 0.7),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(113, "OD")),
        GroundTruth(
            bboxes=[
                LabeledBoundingBox((1, 1), (2, 2), "a"),
                LabeledBoundingBox((3, 3), (4, 4), "a"),
                LabeledBoundingBox((5, 5), (6, 6), "a"),
                LabeledBoundingBox((7, 7), (8, 8), "b"),
            ],
        ),
        Inference(
            bboxes=[
                ScoredLabeledBoundingBox((1.1, 1), (2.1, 2), "a", 0.9),
                ScoredLabeledBoundingBox((3.3, 3), (4.3, 4), "a", 0.8),
                ScoredLabeledBoundingBox((5.5, 5), (6.5, 6), "a", 0.7),
                ScoredLabeledBoundingBox((7.7, 7), (8.7, 8), "b", 1),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(114, "OD")),
        GroundTruth(
            bboxes=[
                LabeledBoundingBox((1, 1), (2, 2), "a"),
                LabeledBoundingBox((3, 3), (4, 4), "b"),
                LabeledBoundingBox((5, 5), (6, 6), "b"),
                LabeledBoundingBox((7, 7), (8, 8), "a"),
            ],
        ),
        Inference(
            bboxes=[
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.6),
                ScoredLabeledBoundingBox((3, 3), (4, 4), "b", 0.5),
                ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.4),
                ScoredLabeledBoundingBox((7, 7), (8, 8), "a", 0.1),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(115, "OD")),
        GroundTruth(
            bboxes=[
                LabeledBoundingBox((1, 1), (2, 2), "a"),
                LabeledBoundingBox((3, 3), (4, 4), "b"),
                LabeledBoundingBox((5, 5), (6, 6), "b"),
                LabeledBoundingBox((7, 7), (8, 8), "a"),
            ],
        ),
        Inference(
            bboxes=[
                ScoredLabeledBoundingBox((0, 0), (1, 1), "a", 1),
                ScoredLabeledBoundingBox((0, 0), (1, 1), "a", 1),
                ScoredLabeledBoundingBox((0, 0), (1, 1), "a", 0.9),
                ScoredLabeledBoundingBox((0, 0), (1, 1), "a", 0.8),
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 1),
                ScoredLabeledBoundingBox((3, 3), (4, 4), "b", 0.9),
                ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.9),
                ScoredLabeledBoundingBox((7, 7), (8, 8), "a", 0.8),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(116, "OD")),
        GroundTruth(
            bboxes=[
                LabeledBoundingBox((1, 1), (2, 2), "a"),
                LabeledBoundingBox((3, 3), (4, 4), "b"),
                LabeledBoundingBox((5, 5), (6, 6), "b"),
                LabeledBoundingBox((7, 7), (8, 8), "a"),
            ],
        ),
        Inference(
            bboxes=[
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 1),
                ScoredLabeledBoundingBox((7, 7), (8, 8), "a", 0.4),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(117, "OD")),
        GroundTruth(
            bboxes=[
                LabeledBoundingBox((1, 1), (2, 2), "a"),
                LabeledBoundingBox((3, 3), (4, 4), "c"),
                LabeledBoundingBox((5, 5), (6, 6), "a"),
                LabeledBoundingBox((7, 7), (8, 8), "a"),
            ],
        ),
        Inference(
            bboxes=[
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0),
                ScoredLabeledBoundingBox((5, 5), (6, 6), "a", 0.9),
                ScoredLabeledBoundingBox((7, 7), (9, 9), "b", 0.9),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(118, "OD")),
        GroundTruth(
            bboxes=[
                LabeledBoundingBox((1, 1), (2, 2), "a"),
                LabeledBoundingBox((3, 3), (4, 4), "c"),
            ],
            ignored_bboxes=[
                LabeledBoundingBox((5, 5), (6, 6), "b"),
                LabeledBoundingBox((7, 7), (8, 8), "a"),
            ],
        ),
        Inference(
            bboxes=[
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.9),
                ScoredLabeledBoundingBox((3, 3), (4, 4), "c", 0.8),
                ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.1),
                ScoredLabeledBoundingBox((7, 7), (8, 8), "a", 1),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(20, "OD")),
        GroundTruth(
            bboxes=[
                LabeledBoundingBox((1, 1), (2, 2), "a"),
                LabeledBoundingBox((3, 3), (4, 4), "a"),
                LabeledBoundingBox((5, 5), (6, 6), "a"),
                LabeledBoundingBox((7, 7), (8, 8), "a"),
                LabeledBoundingBox((11, 11), (12, 12), "b"),
                LabeledBoundingBox((13, 13), (14, 14), "b"),
                LabeledBoundingBox((15, 15), (16, 16), "b"),
                LabeledBoundingBox((17, 17), (18, 18), "b"),
                LabeledBoundingBox((21, 21), (22, 22), "c"),
                LabeledBoundingBox((23, 23), (24, 24), "c"),
                LabeledBoundingBox((25, 25), (26, 26), "c"),
                LabeledBoundingBox((27, 27), (28, 28), "c"),
                LabeledBoundingBox((31, 31), (32, 32), "c"),
                LabeledBoundingBox((33, 33), (34, 34), "c"),
                LabeledBoundingBox((35, 35), (36, 36), "c"),
                LabeledBoundingBox((37, 37), (38, 38), "c"),
                LabeledBoundingBox((41, 41), (42, 42), "c"),
                LabeledBoundingBox((43, 43), (44, 44), "c"),
            ],
            ignored_bboxes=[
                LabeledBoundingBox((2, 2), (3, 3), "b"),
                LabeledBoundingBox((4, 4), (5, 5), "b"),
                LabeledBoundingBox((6, 6), (7, 7), "b"),
                LabeledBoundingBox((8, 8), (9, 9), "b"),
                LabeledBoundingBox((21, 21), (22, 22), "d"),
                LabeledBoundingBox((23, 23), (24, 24), "d"),
            ],
        ),
        Inference(
            bboxes=[
                ScoredLabeledBoundingBox((1, 1), (3, 2), "a", 1),
                ScoredLabeledBoundingBox((3, 3), (5, 4), "a", 1),
                ScoredLabeledBoundingBox((5, 5), (7, 6), "a", 0.9),
                ScoredLabeledBoundingBox((7, 7), (9, 8), "a", 0.8),
                ScoredLabeledBoundingBox((11, 11), (13.01, 12), "b", 0.5),
                ScoredLabeledBoundingBox((13, 13), (15.01, 14), "b", 0.3),
                ScoredLabeledBoundingBox((15, 15), (17.01, 16), "b", 0.1),
                ScoredLabeledBoundingBox((17, 17), (19.01, 18), "b", 0),
                ScoredLabeledBoundingBox((21, 21), (22, 22), "c", 0.9),
                ScoredLabeledBoundingBox((27, 27), (28, 28), "c", 0.6),
                ScoredLabeledBoundingBox((31, 31), (32, 32), "c", 0.5),
                ScoredLabeledBoundingBox((33, 33), (34, 34), "c", 0.4),
                ScoredLabeledBoundingBox((35, 35), (36, 36), "c", 0.3),
                ScoredLabeledBoundingBox((37, 37), (38, 38), "c", 0.2),
                ScoredLabeledBoundingBox((41, 41), (42, 42), "c", 0.1),
                ScoredLabeledBoundingBox((43, 43), (44, 44), "c", 0.1),
                ScoredLabeledBoundingBox((2, 2), (3, 3), "b", 0.8),
                ScoredLabeledBoundingBox((4, 4), (5, 5), "e", 0.8),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(21, "OD")),
        GroundTruth(
            bboxes=[
                LabeledBoundingBox((21, 21), (22, 22), "e"),
                LabeledBoundingBox((23, 23), (24, 24), "e"),
                LabeledBoundingBox((25, 25), (26, 26), "e"),
                LabeledBoundingBox((27, 27), (28, 28), "e"),
                LabeledBoundingBox((31, 31), (32, 32), "e"),
                LabeledBoundingBox((33, 33), (34, 34), "e"),
                LabeledBoundingBox((35, 35), (36, 36), "e"),
                LabeledBoundingBox((37, 37), (38, 38), "e"),
                LabeledBoundingBox((41, 41), (42, 42), "e"),
                LabeledBoundingBox((43, 43), (44, 44), "e"),
            ],
            ignored_bboxes=[
                LabeledBoundingBox((2, 2), (3, 3), "b"),
                LabeledBoundingBox((4, 4), (5, 5), "b"),
                LabeledBoundingBox((6, 6), (7, 7), "b"),
                LabeledBoundingBox((8, 8), (9, 9), "b"),
                LabeledBoundingBox((1, 1), (2, 2), "a"),
                LabeledBoundingBox((3, 3), (4, 4), "a"),
                LabeledBoundingBox((5, 5), (6, 6), "a"),
                LabeledBoundingBox((7, 7), (8, 8), "a"),
            ],
        ),
        Inference(
            bboxes=[
                ScoredLabeledBoundingBox((21, 21), (22, 22), "b", 0.9),
                ScoredLabeledBoundingBox((23, 23), (24, 24), "e", 0.8),
                ScoredLabeledBoundingBox((25, 25), (26, 26), "e", 0.7),
                ScoredLabeledBoundingBox((27, 27), (28, 28), "e", 0.6),
                ScoredLabeledBoundingBox((31, 31), (32, 32), "e", 0.1),
                ScoredLabeledBoundingBox((33, 33), (34, 34), "e", 0),
                ScoredLabeledBoundingBox((6, 6), (7, 7), "b", 0.9),
                ScoredLabeledBoundingBox((8, 8), (9, 9), "b", 0.8),
                ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.7),
                ScoredLabeledBoundingBox((3, 3), (4, 4), "b", 0.6),
                ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.1),
            ],
        ),
    ),
]


EXPECTED_COMPUTE_TEST_SAMPLE_METRICS: List[Tuple[TestSample, TestSampleMetrics]] = [
    (
        TestSample(locator=fake_locator(112, "OD"), metadata={}),
        TestSampleMetrics(
            TP_labels=["a", "b", "c", "d"],
            TP=[
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 1),
                ScoredLabeledBoundingBox((3, 3), (4, 4), "b", 0.9),
                ScoredLabeledBoundingBox((5, 5), (6, 6), "c", 0.8),
                ScoredLabeledBoundingBox((7, 7), (8, 8), "d", 0.7),
            ],
            FP_labels=[],
            FP=[],
            FN_labels=[],
            FN=[],
            Confused_labels=[],
            Confused=[],
            count_TP=4,
            count_FP=0,
            count_FN=0,
            count_Confused=0,
            has_TP=True,
            has_FP=False,
            has_FN=False,
            has_Confused=False,
            max_confidence_above_t=1,
            min_confidence_above_t=0.7,
            thresholds=[
                ScoredClassificationLabel("a", 0.5),
                ScoredClassificationLabel("b", 0.5),
                ScoredClassificationLabel("c", 0.5),
                ScoredClassificationLabel("d", 0.5),
            ],
            inference_labels=["a", "b", "c", "d"],
        ),
    ),
    (
        TestSample(locator=fake_locator(113, "OD"), metadata={}),
        TestSampleMetrics(
            TP_labels=["a"],
            TP=[
                ScoredLabeledBoundingBox((1.1, 1), (2.1, 2), "a", 0.9),
                ScoredLabeledBoundingBox((3.3, 3), (4.3, 4), "a", 0.8),
            ],
            FP_labels=["a", "b"],
            FP=[
                ScoredLabeledBoundingBox((5.5, 5), (6.5, 6), "a", 0.7),
                ScoredLabeledBoundingBox((7.7, 7), (8.7, 8), "b", 1),
            ],
            FN_labels=["a", "b"],
            FN=[
                LabeledBoundingBox((5, 5), (6, 6), "a"),
                LabeledBoundingBox((7, 7), (8, 8), "b"),
            ],
            Confused_labels=[],
            Confused=[],
            count_TP=2,
            count_FP=2,
            count_FN=2,
            count_Confused=0,
            has_TP=True,
            has_FP=True,
            has_FN=True,
            has_Confused=False,
            max_confidence_above_t=1,
            min_confidence_above_t=0.7,
            thresholds=[
                ScoredClassificationLabel("a", 0.5),
                ScoredClassificationLabel("b", 0.5),
            ],
            inference_labels=["a", "b"],
        ),
    ),
    (
        TestSample(locator=fake_locator(114, "OD"), metadata={}),
        TestSampleMetrics(
            TP_labels=["a", "b"],
            TP=[
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.6),
                ScoredLabeledBoundingBox((3, 3), (4, 4), "b", 0.5),
            ],
            FP_labels=[],
            FP=[],
            FN_labels=["a", "b"],
            FN=[
                LabeledBoundingBox((7, 7), (8, 8), "a"),
                LabeledBoundingBox((5, 5), (6, 6), "b"),
            ],
            Confused_labels=[],
            Confused=[],
            count_TP=2,
            count_FP=0,
            count_FN=2,
            count_Confused=0,
            has_TP=True,
            has_FP=False,
            has_FN=True,
            has_Confused=False,
            max_confidence_above_t=0.6,
            min_confidence_above_t=0.5,
            thresholds=[
                ScoredClassificationLabel("a", 0.5),
                ScoredClassificationLabel("b", 0.5),
            ],
            inference_labels=["a", "b"],
        ),
    ),
    (
        TestSample(locator=fake_locator(115, "OD"), metadata={}),
        TestSampleMetrics(
            TP_labels=["a", "b"],
            TP=[
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 1),
                ScoredLabeledBoundingBox((7, 7), (8, 8), "a", 0.8),
                ScoredLabeledBoundingBox((3, 3), (4, 4), "b", 0.9),
                ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.9),
            ],
            FP_labels=["a"],
            FP=[
                ScoredLabeledBoundingBox((0, 0), (1, 1), "a", 1),
                ScoredLabeledBoundingBox((0, 0), (1, 1), "a", 1),
                ScoredLabeledBoundingBox((0, 0), (1, 1), "a", 0.9),
                ScoredLabeledBoundingBox((0, 0), (1, 1), "a", 0.8),
            ],
            FN_labels=[],
            FN=[],
            Confused_labels=[],
            Confused=[],
            count_TP=4,
            count_FP=4,
            count_FN=0,
            count_Confused=0,
            has_TP=True,
            has_FP=True,
            has_FN=False,
            has_Confused=False,
            max_confidence_above_t=1,
            min_confidence_above_t=0.8,
            thresholds=[
                ScoredClassificationLabel("a", 0.5),
                ScoredClassificationLabel("b", 0.5),
            ],
            inference_labels=["a", "b"],
        ),
    ),
    (
        TestSample(locator=fake_locator(116, "OD"), metadata={}),
        TestSampleMetrics(
            TP_labels=["a"],
            TP=[ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 1)],
            FP_labels=[],
            FP=[],
            FN_labels=["a", "b"],
            FN=[
                LabeledBoundingBox((3, 3), (4, 4), "b"),
                LabeledBoundingBox((5, 5), (6, 6), "b"),
                LabeledBoundingBox((7, 7), (8, 8), "a"),
            ],
            Confused_labels=[],
            Confused=[],
            count_TP=1,
            count_FP=0,
            count_FN=3,
            count_Confused=0,
            has_TP=True,
            has_FP=False,
            has_FN=True,
            has_Confused=False,
            max_confidence_above_t=1,
            min_confidence_above_t=1,
            thresholds=[ScoredClassificationLabel("a", 0.5)],
            inference_labels=["a"],
        ),
    ),
    (
        TestSample(locator=fake_locator(117, "OD"), metadata={}),
        TestSampleMetrics(
            TP_labels=["a"],
            TP=[ScoredLabeledBoundingBox((5, 5), (6, 6), "a", 0.9)],
            FP_labels=["b"],
            FP=[ScoredLabeledBoundingBox((7, 7), (9, 9), "b", 0.9)],
            FN_labels=["a", "c"],
            FN=[
                LabeledBoundingBox((7, 7), (8, 8), "a"),
                LabeledBoundingBox((3, 3), (4, 4), "c"),
                LabeledBoundingBox((1, 1), (2, 2), "a"),
            ],
            Confused_labels=[],
            Confused=[],
            count_TP=1,
            count_FP=1,
            count_FN=3,
            count_Confused=0,
            has_TP=True,
            has_FP=True,
            has_FN=True,
            has_Confused=False,
            max_confidence_above_t=0.9,
            min_confidence_above_t=0.9,
            thresholds=[ScoredClassificationLabel("a", 0.5)],
            inference_labels=["a", "b"],
        ),
    ),
    (
        TestSample(locator=fake_locator(118, "OD"), metadata={}),
        TestSampleMetrics(
            TP_labels=["a", "c"],
            TP=[
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.9),
                ScoredLabeledBoundingBox((3, 3), (4, 4), "c", 0.8),
            ],
            FP_labels=[],
            FP=[],
            FN_labels=[],
            FN=[],
            Confused_labels=[],
            Confused=[],
            count_TP=2,
            count_FP=0,
            count_FN=0,
            count_Confused=0,
            has_TP=True,
            has_FP=False,
            has_FN=False,
            has_Confused=False,
            max_confidence_above_t=0.9,
            min_confidence_above_t=0.8,
            thresholds=[
                ScoredClassificationLabel("a", 0.5),
                ScoredClassificationLabel("c", 0.5),
            ],
            inference_labels=["a", "c"],
        ),
    ),
    (
        TestSample(locator=fake_locator(20, "OD"), metadata={}),
        TestSampleMetrics(
            TP_labels=["a", "c"],
            TP=[
                ScoredLabeledBoundingBox((1, 1), (3, 2), "a", 1),
                ScoredLabeledBoundingBox((3, 3), (5, 4), "a", 1),
                ScoredLabeledBoundingBox((5, 5), (7, 6), "a", 0.9),
                ScoredLabeledBoundingBox((7, 7), (9, 8), "a", 0.8),
                ScoredLabeledBoundingBox((21, 21), (22, 22), "c", 0.9),
                ScoredLabeledBoundingBox((27, 27), (28, 28), "c", 0.6),
                ScoredLabeledBoundingBox((31, 31), (32, 32), "c", 0.5),
            ],
            FP_labels=["b", "e"],
            FP=[
                ScoredLabeledBoundingBox((11, 11), (13.01, 12), "b", 0.5),
                ScoredLabeledBoundingBox((4, 4), (5, 5), "e", 0.8),
            ],
            FN_labels=["b", "c"],
            FN=[
                LabeledBoundingBox((11, 11), (12, 12), "b"),
                LabeledBoundingBox((13, 13), (14, 14), "b"),
                LabeledBoundingBox((15, 15), (16, 16), "b"),
                LabeledBoundingBox((17, 17), (18, 18), "b"),
                LabeledBoundingBox((23, 23), (24, 24), "c"),
                LabeledBoundingBox((25, 25), (26, 26), "c"),
                LabeledBoundingBox((33, 33), (34, 34), "c"),
                LabeledBoundingBox((35, 35), (36, 36), "c"),
                LabeledBoundingBox((37, 37), (38, 38), "c"),
                LabeledBoundingBox((41, 41), (42, 42), "c"),
                LabeledBoundingBox((43, 43), (44, 44), "c"),
            ],
            Confused_labels=[],
            Confused=[],
            count_TP=7,
            count_FP=2,
            count_FN=11,
            count_Confused=0,
            has_TP=True,
            has_FP=True,
            has_FN=True,
            has_Confused=False,
            max_confidence_above_t=1,
            min_confidence_above_t=0.5,
            thresholds=[
                ScoredClassificationLabel("a", 0.5),
                ScoredClassificationLabel("c", 0.5),
                ScoredClassificationLabel("b", 0.5),
            ],
            inference_labels=["a", "b", "c", "e"],
        ),
    ),
    (
        TestSample(locator=fake_locator(21, "OD"), metadata={}),
        TestSampleMetrics(
            TP_labels=["e"],
            TP=[
                ScoredLabeledBoundingBox((23, 23), (24, 24), "e", 0.8),
                ScoredLabeledBoundingBox((25, 25), (26, 26), "e", 0.7),
                ScoredLabeledBoundingBox((27, 27), (28, 28), "e", 0.6),
            ],
            FP_labels=["b"],
            FP=[
                ScoredLabeledBoundingBox((21, 21), (22, 22), "b", 0.9),
                ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.7),
                ScoredLabeledBoundingBox((3, 3), (4, 4), "b", 0.6),
            ],
            FN_labels=["e"],
            FN=[
                LabeledBoundingBox((21, 21), (22, 22), "e"),
                LabeledBoundingBox((35, 35), (36, 36), "e"),
                LabeledBoundingBox((37, 37), (38, 38), "e"),
                LabeledBoundingBox((41, 41), (42, 42), "e"),
                LabeledBoundingBox((43, 43), (44, 44), "e"),
                LabeledBoundingBox((31, 31), (32, 32), "e"),
                LabeledBoundingBox((33, 33), (34, 34), "e"),
            ],
            Confused_labels=["b"],
            Confused=[
                ScoredLabeledBoundingBox((21, 21), (22, 22), "b", 0.9),
            ],
            count_TP=3,
            count_FP=3,
            count_FN=7,
            count_Confused=1,
            has_TP=True,
            has_FP=True,
            has_FN=True,
            has_Confused=True,
            max_confidence_above_t=0.9,
            min_confidence_above_t=0.6,
            thresholds=[ScoredClassificationLabel("e", 0.5)],
            inference_labels=["b", "e"],
        ),
    ),
]


EXPECTED_COMPUTE_TEST_CASE_METRICS = TestCaseMetrics(
    PerClass=[
        ClassMetricsPerTestCase(
            Class="a",
            nImages=8,
            Threshold=0.5,
            Objects=18,
            Inferences=18,
            TP=13,
            FN=5,
            FP=5,
            Precision=13 / 18,
            Recall=13 / 18,
            F1=13 / 18,
            AP=128 / 189,
        ),
        ClassMetricsPerTestCase(
            Class="b",
            nImages=8,
            Threshold=0.5,
            Objects=12,
            Inferences=10,
            TP=4,
            FN=8,
            FP=6,
            Precision=2 / 5,
            Recall=1 / 3,
            F1=4 / 11,
            AP=53 / 264,
        ),
        ClassMetricsPerTestCase(
            Class="c",
            nImages=4,
            Threshold=0.5,
            Objects=13,
            Inferences=5,
            TP=5,
            FN=8,
            FP=0,
            Precision=1,
            Recall=5 / 13,
            F1=5 / 9,
            AP=10 / 13,
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
            nImages=2,
            Threshold=0.5,
            Objects=10,
            Inferences=4,
            TP=3,
            FN=7,
            FP=1,
            Precision=0.75,
            Recall=0.3,
            F1=3 / 7,
            AP=5 / 12,
        ),
    ],
    Objects=54,
    Inferences=38,
    TP=26,
    FN=28,
    FP=12,
    macro_Precision=697 / 900,
    macro_Recall=1603 / 2925,
    macro_F1=851 / 1386,
    mean_AP=132493 / 216216,
)


EXPECTED_COMPUTE_TEST_CASE_PLOTS: List[Plot] = [
    CurvePlot(
        title="F1-Score vs. Confidence Threshold Per Class",
        x_label="Confidence Threshold",
        y_label="F1-Score",
        curves=[
            Curve(
                x=[0, 0.1, 0.4, 0.6, 0.7, 0.8, 0.9, 1],
                y=[32 / 39, 15 / 19, 28 / 37, 13 / 18, 24 / 35, 12 / 17, 0.6, 0.4],
                label="a",
            ),
            Curve(
                x=[0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1],
                y=[10 / 27, 5 / 13, 5 / 12, 10 / 23, 4 / 11, 0.3, 6 / 19, 1 / 3, 0],
                label="b",
            ),
            Curve(
                x=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9],
                y=[20 / 23, 16 / 21, 14 / 20, 12 / 19, 5 / 9, 8 / 17, 3 / 8, 1 / 7],
                label="c",
            ),
            Curve(x=[0, 0.1, 0.6, 0.7, 0.8], y=[5 / 8, 8 / 15, 3 / 7, 4 / 13, 1 / 6], label="e"),
        ],
        x_config=None,
        y_config=None,
    ),
    CurvePlot(
        title="Precision vs. Recall Per Class",
        x_label="Recall",
        y_label="Precision",
        curves=[
            Curve(
                x=[8 / 9, 5 / 6, 7 / 9, 13 / 18, 6 / 9, 0.5, 5 / 18, 0],
                y=[16 / 21, 0.75, 14 / 19, 13 / 18, 0.75, 0.75, 5 / 7, 5 / 7],
                label="a",
            ),
            Curve(x=[5 / 12, 1 / 3, 0.25, 0], y=[5 / 11, 0.4, 0.5, 0], label="b"),
            Curve(
                x=[10 / 13, 8 / 13, 7 / 13, 6 / 13, 5 / 13, 4 / 13, 3 / 13, 1 / 13, 0],
                y=[1, 1, 1, 1, 1, 1, 1, 1, 1],
                label="c",
            ),
            Curve(x=[1, 0], y=[1, 1], label="d"),
            Curve(x=[0.5, 0.4, 0.3, 0.2, 0.1, 0], y=[5 / 6, 0.8, 0.75, 6 / 9, 0.5, 0.5], label="e"),
        ],
        x_config=None,
        y_config=None,
    ),
    ConfusionMatrix(
        title="Confusion Matrix",
        labels=["a", "b", "c", "d", "e"],
        matrix=[[13, 0, 0, 0, 0], [0, 4, 0, 0, 0], [0, 0, 5, 0, 0], [0, 0, 0, 1, 0], [0, 1, 0, 0, 3]],
        x_label="Predicted",
        y_label="Actual",
    ),
]


def assert_test_case_metrics_equals_expected(
    metrics: TestCaseMetrics,
    other_metrics: TestCaseMetrics,
) -> None:
    assert len(metrics.PerClass) == len(other_metrics.PerClass)
    for pc_metric, expected_pc_metric in zip(metrics.PerClass, other_metrics.PerClass):
        assert pc_metric.Class == expected_pc_metric.Class
        assert pc_metric.nImages == expected_pc_metric.nImages
        assert pc_metric.Threshold == expected_pc_metric.Threshold
        assert pc_metric.Objects == expected_pc_metric.Objects
        assert pc_metric.Inferences == expected_pc_metric.Inferences
        assert pc_metric.TP == expected_pc_metric.TP
        assert pc_metric.FN == expected_pc_metric.FN
        assert pc_metric.FP == expected_pc_metric.FP
        assert pytest.approx(pc_metric.Precision, abs=1e-12) == expected_pc_metric.Precision
        assert pytest.approx(pc_metric.Recall, abs=1e-12) == expected_pc_metric.Recall
        assert pytest.approx(pc_metric.F1, abs=1e-12) == expected_pc_metric.F1
        assert pytest.approx(pc_metric.AP, abs=1e-12) == expected_pc_metric.AP

    assert metrics.Objects == other_metrics.Objects
    assert metrics.Inferences == other_metrics.Inferences
    assert metrics.TP == other_metrics.TP
    assert metrics.FN == other_metrics.FN
    assert metrics.FP == other_metrics.FP
    assert pytest.approx(metrics.macro_Precision, abs=1e-12) == other_metrics.macro_Precision
    assert pytest.approx(metrics.macro_Recall, abs=1e-12) == other_metrics.macro_Recall
    assert pytest.approx(metrics.macro_F1, abs=1e-12) == other_metrics.macro_F1
    assert pytest.approx(metrics.mean_AP, abs=1e-12) == other_metrics.mean_AP


def assert_curves(
    curves: List[Curve],
    expected: List[Curve],
) -> None:
    assert len(curves) == len(expected)
    for curve, expectation in zip(curves, expected):
        assert curve.label == expectation.label
        assert len(curve.x) == len(expectation.x)
        assert sum(abs(a - b) for a, b in zip(curve.x, expectation.x)) < 1e-12
        assert len(curve.y) == len(expectation.y)
        assert sum(abs(a - b) for a, b in zip(curve.y, expectation.y)) < 1e-12


def assert_test_case_plots_equals_expected(
    plots: List[Plot],
    other_plots: List[Plot],
) -> None:
    assert len(plots) == len(other_plots)
    # check curve plots
    for plot, expected in zip(plots[:2], other_plots[:2]):
        assert plot.title == expected.title
        assert plot.x_label == expected.x_label
        assert plot.y_label == expected.y_label
        assert_curves(plot.curves, expected.curves)
        assert plot.x_config == expected.x_config
        assert plot.y_config == expected.y_config

    # check confusion matrix
    assert plots[2] == other_plots[2]


@pytest.mark.metrics
def test__object_detection__multiclass_evaluator__fixed() -> None:
    from kolena._experimental.object_detection import ObjectDetectionEvaluator

    config = ThresholdConfiguration(
        threshold_strategy=ThresholdStrategy.FIXED_05,
        iou_threshold=0.5,
        min_confidence_score=0,
        with_class_level_metrics=True,
    )
    eval = ObjectDetectionEvaluator(configurations=[config])

    test_sample_metrics = eval.compute_test_sample_metrics(
        test_case=TEST_CASE,
        inferences=TEST_DATA,
        configuration=config,
    )

    assert len(eval.evaluator.threshold_cache) == 0  # empty because not f1 optimal config
    assert len(eval.evaluator.matchings_by_test_case) != 0
    assert len(eval.evaluator.matchings_by_test_case[TEST_CASE.name]) == len(TEST_DATA)
    assert test_sample_metrics == EXPECTED_COMPUTE_TEST_SAMPLE_METRICS

    # test case metrics, which will populate the locators cache
    assert len(eval.evaluator.locators_by_test_case) == 0

    test_case_metrics = eval.compute_test_case_metrics(
        test_case=TEST_CASE,
        inferences=TEST_DATA,
        metrics=[pair[1] for pair in EXPECTED_COMPUTE_TEST_SAMPLE_METRICS],
        configuration=config,
    )

    assert len(eval.evaluator.locators_by_test_case) == 1  # cache contains locators for one test case
    assert len(eval.evaluator.locators_by_test_case[TEST_CASE.name]) == len(TEST_DATA)
    assert_test_case_metrics_equals_expected(test_case_metrics, EXPECTED_COMPUTE_TEST_CASE_METRICS)

    # test case plots only use the cached values
    plots = eval.compute_test_case_plots(
        test_case=TEST_CASE,
        inferences=[],
        metrics=[],
        configuration=config,
    )
    assert_test_case_plots_equals_expected(plots, EXPECTED_COMPUTE_TEST_CASE_PLOTS)

    # test suite metrics - one
    test_suite_metrics = eval.compute_test_suite_metrics(
        test_suite=TEST_SUITE,
        metrics=[(TEST_CASE, EXPECTED_COMPUTE_TEST_CASE_METRICS)],
        configuration=config,
    )
    assert test_suite_metrics == TestSuiteMetrics(n_images=9, mean_AP=132493 / 216216)

    # test suite metrics - two
    test_suite_metrics_dup = eval.compute_test_suite_metrics(
        test_suite=TEST_SUITE,
        metrics=[
            (TEST_CASE, EXPECTED_COMPUTE_TEST_CASE_METRICS),
            (TEST_CASE, EXPECTED_COMPUTE_TEST_CASE_METRICS),
        ],
        configuration=config,
    )
    assert test_suite_metrics_dup == TestSuiteMetrics(n_images=9, mean_AP=132493 / 216216)
