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
import uuid
from typing import Dict
from typing import Tuple

import pytest

from .test_evaluator_single_class_test_sample_metrics import EXPECTED_COMPUTE_TEST_SAMPLE_METRICS
from .test_evaluator_single_class_test_sample_metrics import TEST_CONFIGURATIONS
from .test_evaluator_single_class_test_sample_metrics import TEST_DATA
from .test_evaluator_single_class_test_sample_metrics import TEST_PARAMS
from kolena._experimental.object_detection import ObjectDetectionEvaluator
from kolena._experimental.object_detection import TestCase
from kolena._experimental.object_detection.workflow import TestCaseMetricsSingleClass
from tests.integration.helper import with_test_prefix


# evaluator_configuration, test_name -> test_case_metrics
EXPECTED_COMPUTE_TEST_CASE_METRICS: Dict[Tuple[str, str], TestCaseMetricsSingleClass] = {
    ("Threshold: Fixed(0.3), IoU: 0.3, confidence ≥ 0.0", "nothing"): TestCaseMetricsSingleClass(
        Objects=0,
        Inferences=0,
        TP=0,
        FN=0,
        FP=0,
        Precision=0.0,
        Recall=0.0,
        F1=0.0,
        AP=0.0,
    ),
    ("Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.0", "nothing"): TestCaseMetricsSingleClass(
        Objects=0,
        Inferences=0,
        TP=0,
        FN=0,
        FP=0,
        Precision=0.0,
        Recall=0.0,
        F1=0.0,
        AP=0.0,
    ),
    ("Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.3", "nothing"): TestCaseMetricsSingleClass(
        Objects=0,
        Inferences=0,
        TP=0,
        FN=0,
        FP=0,
        Precision=0.0,
        Recall=0.0,
        F1=0.0,
        AP=0.0,
    ),
    ("Threshold: F1-Optimal, IoU: 0.5, confidence ≥ 0.1", "nothing"): TestCaseMetricsSingleClass(
        Objects=0,
        Inferences=0,
        TP=0,
        FN=0,
        FP=0,
        Precision=0.0,
        Recall=0.0,
        F1=0.0,
        AP=0.0,
    ),
    ("Threshold: Fixed(0.3), IoU: 0.3, confidence ≥ 0.0", "no inferences"): TestCaseMetricsSingleClass(
        Objects=1,
        Inferences=0,
        TP=0,
        FN=1,
        FP=0,
        Precision=0.0,
        Recall=0.0,
        F1=0.0,
        AP=0.0,
    ),
    ("Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.0", "no inferences"): TestCaseMetricsSingleClass(
        Objects=1,
        Inferences=0,
        TP=0,
        FN=1,
        FP=0,
        Precision=0.0,
        Recall=0.0,
        F1=0.0,
        AP=0.0,
    ),
    ("Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.3", "no inferences"): TestCaseMetricsSingleClass(
        Objects=1,
        Inferences=0,
        TP=0,
        FN=1,
        FP=0,
        Precision=0.0,
        Recall=0.0,
        F1=0.0,
        AP=0.0,
    ),
    ("Threshold: F1-Optimal, IoU: 0.5, confidence ≥ 0.1", "no inferences"): TestCaseMetricsSingleClass(
        Objects=1,
        Inferences=0,
        TP=0,
        FN=1,
        FP=0,
        Precision=0.0,
        Recall=0.0,
        F1=0.0,
        AP=0.0,
    ),
    ("Threshold: Fixed(0.3), IoU: 0.3, confidence ≥ 0.0", "no ground truths"): TestCaseMetricsSingleClass(
        Objects=0,
        Inferences=1,
        TP=0,
        FN=0,
        FP=1,
        Precision=0.0,
        Recall=0.0,
        F1=0.0,
        AP=0.0,
    ),
    ("Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.0", "no ground truths"): TestCaseMetricsSingleClass(
        Objects=0,
        Inferences=1,
        TP=0,
        FN=0,
        FP=1,
        Precision=0.0,
        Recall=0.0,
        F1=0.0,
        AP=0.0,
    ),
    ("Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.3", "no ground truths"): TestCaseMetricsSingleClass(
        Objects=0,
        Inferences=1,
        TP=0,
        FN=0,
        FP=1,
        Precision=0.0,
        Recall=0.0,
        F1=0.0,
        AP=0.0,
    ),
    ("Threshold: F1-Optimal, IoU: 0.5, confidence ≥ 0.1", "no ground truths"): TestCaseMetricsSingleClass(
        Objects=0,
        Inferences=1,
        TP=0,
        FN=0,
        FP=1,
        Precision=0.0,
        Recall=0.0,
        F1=0.0,
        AP=0.0,
    ),
    (
        "Threshold: Fixed(0.3), IoU: 0.3, confidence ≥ 0.0",
        "iou=1 and different labels and max confidence",
    ): TestCaseMetricsSingleClass(Objects=1, Inferences=1, TP=1, FN=0, FP=0, Precision=1.0, Recall=1.0, F1=1.0, AP=0.0),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.0",
        "iou=1 and different labels and max confidence",
    ): TestCaseMetricsSingleClass(Objects=1, Inferences=1, TP=1, FN=0, FP=0, Precision=1.0, Recall=1.0, F1=1.0, AP=0.0),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.3",
        "iou=1 and different labels and max confidence",
    ): TestCaseMetricsSingleClass(Objects=1, Inferences=1, TP=1, FN=0, FP=0, Precision=1.0, Recall=1.0, F1=1.0, AP=0.0),
    (
        "Threshold: F1-Optimal, IoU: 0.5, confidence ≥ 0.1",
        "iou=1 and different labels and max confidence",
    ): TestCaseMetricsSingleClass(Objects=1, Inferences=1, TP=1, FN=0, FP=0, Precision=1.0, Recall=1.0, F1=1.0, AP=0.0),
    ("Threshold: Fixed(0.3), IoU: 0.3, confidence ≥ 0.0", "iou=0 and same labels"): TestCaseMetricsSingleClass(
        Objects=1,
        Inferences=1,
        TP=0,
        FN=1,
        FP=1,
        Precision=0.0,
        Recall=0.0,
        F1=0.0,
        AP=0.0,
    ),
    ("Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.0", "iou=0 and same labels"): TestCaseMetricsSingleClass(
        Objects=1,
        Inferences=1,
        TP=0,
        FN=1,
        FP=1,
        Precision=0.0,
        Recall=0.0,
        F1=0.0,
        AP=0.0,
    ),
    ("Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.3", "iou=0 and same labels"): TestCaseMetricsSingleClass(
        Objects=1,
        Inferences=1,
        TP=0,
        FN=1,
        FP=1,
        Precision=0.0,
        Recall=0.0,
        F1=0.0,
        AP=0.0,
    ),
    ("Threshold: F1-Optimal, IoU: 0.5, confidence ≥ 0.1", "iou=0 and same labels"): TestCaseMetricsSingleClass(
        Objects=1,
        Inferences=1,
        TP=0,
        FN=1,
        FP=1,
        Precision=0.0,
        Recall=0.0,
        F1=0.0,
        AP=0.0,
    ),
    (
        "Threshold: Fixed(0.3), IoU: 0.3, confidence ≥ 0.0",
        "iou=0.33 and same labels but 0 confidence",
    ): TestCaseMetricsSingleClass(Objects=1, Inferences=0, TP=0, FN=1, FP=0, Precision=0.0, Recall=0.0, F1=0.0, AP=0.0),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.0",
        "iou=0.33 and same labels but 0 confidence",
    ): TestCaseMetricsSingleClass(Objects=1, Inferences=0, TP=0, FN=1, FP=0, Precision=0.0, Recall=0.0, F1=0.0, AP=0.0),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.3",
        "iou=0.33 and same labels but 0 confidence",
    ): TestCaseMetricsSingleClass(Objects=1, Inferences=0, TP=0, FN=1, FP=0, Precision=0.0, Recall=0.0, F1=0.0, AP=0.0),
    (
        "Threshold: F1-Optimal, IoU: 0.5, confidence ≥ 0.1",
        "iou=0.33 and same labels but 0 confidence",
    ): TestCaseMetricsSingleClass(Objects=1, Inferences=0, TP=0, FN=1, FP=0, Precision=0.0, Recall=0.0, F1=0.0, AP=0.0),
    (
        "Threshold: Fixed(0.3), IoU: 0.3, confidence ≥ 0.0",
        "iou=0.33 and same labels but 0.5 confidence",
    ): TestCaseMetricsSingleClass(Objects=1, Inferences=1, TP=0, FN=1, FP=1, Precision=0.0, Recall=0.0, F1=0.0, AP=0.0),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.0",
        "iou=0.33 and same labels but 0.5 confidence",
    ): TestCaseMetricsSingleClass(Objects=1, Inferences=1, TP=0, FN=1, FP=1, Precision=0.0, Recall=0.0, F1=0.0, AP=0.0),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.3",
        "iou=0.33 and same labels but 0.5 confidence",
    ): TestCaseMetricsSingleClass(Objects=1, Inferences=1, TP=0, FN=1, FP=1, Precision=0.0, Recall=0.0, F1=0.0, AP=0.0),
    (
        "Threshold: F1-Optimal, IoU: 0.5, confidence ≥ 0.1",
        "iou=0.33 and same labels but 0.5 confidence",
    ): TestCaseMetricsSingleClass(Objects=1, Inferences=1, TP=0, FN=1, FP=1, Precision=0.0, Recall=0.0, F1=0.0, AP=0.0),
    (
        "Threshold: Fixed(0.3), IoU: 0.3, confidence ≥ 0.0",
        "iou=0.33 and same labels but 0.99 confidence",
    ): TestCaseMetricsSingleClass(Objects=1, Inferences=1, TP=0, FN=1, FP=1, Precision=0.0, Recall=0.0, F1=0.0, AP=0.0),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.0",
        "iou=0.33 and same labels but 0.99 confidence",
    ): TestCaseMetricsSingleClass(Objects=1, Inferences=1, TP=0, FN=1, FP=1, Precision=0.0, Recall=0.0, F1=0.0, AP=0.0),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.3",
        "iou=0.33 and same labels but 0.99 confidence",
    ): TestCaseMetricsSingleClass(Objects=1, Inferences=1, TP=0, FN=1, FP=1, Precision=0.0, Recall=0.0, F1=0.0, AP=0.0),
    (
        "Threshold: F1-Optimal, IoU: 0.5, confidence ≥ 0.1",
        "iou=0.33 and same labels but 0.99 confidence",
    ): TestCaseMetricsSingleClass(Objects=1, Inferences=1, TP=0, FN=1, FP=1, Precision=0.0, Recall=0.0, F1=0.0, AP=0.0),
    (
        "Threshold: Fixed(0.3), IoU: 0.3, confidence ≥ 0.0",
        "iou=0.5 and same labels but 0 confidence",
    ): TestCaseMetricsSingleClass(Objects=1, Inferences=0, TP=0, FN=1, FP=0, Precision=0.0, Recall=0.0, F1=0.0, AP=0.0),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.0",
        "iou=0.5 and same labels but 0 confidence",
    ): TestCaseMetricsSingleClass(Objects=1, Inferences=0, TP=0, FN=1, FP=0, Precision=0.0, Recall=0.0, F1=0.0, AP=0.0),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.3",
        "iou=0.5 and same labels but 0 confidence",
    ): TestCaseMetricsSingleClass(Objects=1, Inferences=0, TP=0, FN=1, FP=0, Precision=0.0, Recall=0.0, F1=0.0, AP=0.0),
    (
        "Threshold: F1-Optimal, IoU: 0.5, confidence ≥ 0.1",
        "iou=0.5 and same labels but 0 confidence",
    ): TestCaseMetricsSingleClass(Objects=1, Inferences=0, TP=0, FN=1, FP=0, Precision=0.0, Recall=0.0, F1=0.0, AP=0.0),
    (
        "Threshold: Fixed(0.3), IoU: 0.3, confidence ≥ 0.0",
        "iou=0.5 and same labels but 0.5 confidence",
    ): TestCaseMetricsSingleClass(
        Objects=1,
        Inferences=1,
        TP=1,
        FN=0,
        FP=0,
        Precision=1.0,
        Recall=1.0,
        F1=1.0,
        AP=0.0,
    ),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.0",
        "iou=0.5 and same labels but 0.5 confidence",
    ): TestCaseMetricsSingleClass(
        Objects=1,
        Inferences=1,
        TP=1,
        FN=0,
        FP=0,
        Precision=1.0,
        Recall=1.0,
        F1=1.0,
        AP=0.0,
    ),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.3",
        "iou=0.5 and same labels but 0.5 confidence",
    ): TestCaseMetricsSingleClass(
        Objects=1,
        Inferences=1,
        TP=1,
        FN=0,
        FP=0,
        Precision=1.0,
        Recall=1.0,
        F1=1.0,
        AP=0.0,
    ),
    (
        "Threshold: F1-Optimal, IoU: 0.5, confidence ≥ 0.1",
        "iou=0.5 and same labels but 0.5 confidence",
    ): TestCaseMetricsSingleClass(
        Objects=1,
        Inferences=1,
        TP=1,
        FN=0,
        FP=0,
        Precision=1.0,
        Recall=1.0,
        F1=1.0,
        AP=0.0,
    ),
    (
        "Threshold: Fixed(0.3), IoU: 0.3, confidence ≥ 0.0",
        "iou=0.5 and same labels but 0.99 confidence",
    ): TestCaseMetricsSingleClass(
        Objects=1,
        Inferences=1,
        TP=1,
        FN=0,
        FP=0,
        Precision=1.0,
        Recall=1.0,
        F1=1.0,
        AP=0.0,
    ),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.0",
        "iou=0.5 and same labels but 0.99 confidence",
    ): TestCaseMetricsSingleClass(
        Objects=1,
        Inferences=1,
        TP=1,
        FN=0,
        FP=0,
        Precision=1.0,
        Recall=1.0,
        F1=1.0,
        AP=0.0,
    ),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.3",
        "iou=0.5 and same labels but 0.99 confidence",
    ): TestCaseMetricsSingleClass(
        Objects=1,
        Inferences=1,
        TP=1,
        FN=0,
        FP=0,
        Precision=1.0,
        Recall=1.0,
        F1=1.0,
        AP=0.0,
    ),
    (
        "Threshold: F1-Optimal, IoU: 0.5, confidence ≥ 0.1",
        "iou=0.5 and same labels but 0.99 confidence",
    ): TestCaseMetricsSingleClass(
        Objects=1,
        Inferences=1,
        TP=1,
        FN=0,
        FP=0,
        Precision=1.0,
        Recall=1.0,
        F1=1.0,
        AP=0.0,
    ),
    (
        "Threshold: Fixed(0.3), IoU: 0.3, confidence ≥ 0.0",
        "multiple bboxes in an image, perfect match",
    ): TestCaseMetricsSingleClass(
        Objects=4,
        Inferences=4,
        TP=4,
        FN=0,
        FP=0,
        Precision=1.0,
        Recall=1.0,
        F1=1.0,
        AP=0.0,
    ),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.0",
        "multiple bboxes in an image, perfect match",
    ): TestCaseMetricsSingleClass(
        Objects=4,
        Inferences=4,
        TP=4,
        FN=0,
        FP=0,
        Precision=1.0,
        Recall=1.0,
        F1=1.0,
        AP=0.0,
    ),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.3",
        "multiple bboxes in an image, perfect match",
    ): TestCaseMetricsSingleClass(
        Objects=4,
        Inferences=4,
        TP=4,
        FN=0,
        FP=0,
        Precision=1.0,
        Recall=1.0,
        F1=1.0,
        AP=0.0,
    ),
    (
        "Threshold: F1-Optimal, IoU: 0.5, confidence ≥ 0.1",
        "multiple bboxes in an image, perfect match",
    ): TestCaseMetricsSingleClass(
        Objects=4,
        Inferences=4,
        TP=4,
        FN=0,
        FP=0,
        Precision=1.0,
        Recall=1.0,
        F1=1.0,
        AP=0.0,
    ),
    (
        "Threshold: Fixed(0.3), IoU: 0.3, confidence ≥ 0.0",
        "multiple bboxes in an image, varied iou",
    ): TestCaseMetricsSingleClass(
        Objects=4,
        Inferences=4,
        TP=3,
        FN=1,
        FP=1,
        Precision=0.75,
        Recall=0.75,
        F1=0.75,
        AP=0.0,
    ),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.0",
        "multiple bboxes in an image, varied iou",
    ): TestCaseMetricsSingleClass(
        Objects=4,
        Inferences=4,
        TP=2,
        FN=2,
        FP=2,
        Precision=0.5,
        Recall=0.5,
        F1=0.5,
        AP=0.0,
    ),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.3",
        "multiple bboxes in an image, varied iou",
    ): TestCaseMetricsSingleClass(
        Objects=4,
        Inferences=4,
        TP=2,
        FN=2,
        FP=2,
        Precision=0.5,
        Recall=0.5,
        F1=0.5,
        AP=0.0,
    ),
    (
        "Threshold: F1-Optimal, IoU: 0.5, confidence ≥ 0.1",
        "multiple bboxes in an image, varied iou",
    ): TestCaseMetricsSingleClass(
        Objects=4,
        Inferences=4,
        TP=2,
        FN=2,
        FP=2,
        Precision=0.5,
        Recall=0.5,
        F1=0.5,
        AP=0.0,
    ),
    (
        "Threshold: Fixed(0.3), IoU: 0.3, confidence ≥ 0.0",
        "multiple bboxes in an image, varied confidence",
    ): TestCaseMetricsSingleClass(
        Objects=4,
        Inferences=3,
        TP=3,
        FN=1,
        FP=0,
        Precision=1.0,
        Recall=0.75,
        F1=0.8571428571428571,
        AP=1.0,
    ),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.0",
        "multiple bboxes in an image, varied confidence",
    ): TestCaseMetricsSingleClass(
        Objects=4,
        Inferences=2,
        TP=2,
        FN=2,
        FP=0,
        Precision=1.0,
        Recall=0.5,
        F1=0.6666666666666666,
        AP=1.0,
    ),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.3",
        "multiple bboxes in an image, varied confidence",
    ): TestCaseMetricsSingleClass(
        Objects=4,
        Inferences=2,
        TP=2,
        FN=2,
        FP=0,
        Precision=1.0,
        Recall=0.5,
        F1=0.6666666666666666,
        AP=0.75,
    ),
    (
        "Threshold: F1-Optimal, IoU: 0.5, confidence ≥ 0.1",
        "multiple bboxes in an image, varied confidence",
    ): TestCaseMetricsSingleClass(
        Objects=4,
        Inferences=3,
        TP=3,
        FN=1,
        FP=0,
        Precision=1.0,
        Recall=0.75,
        F1=0.8571428571428571,
        AP=0.75,
    ),
    (
        "Threshold: Fixed(0.3), IoU: 0.3, confidence ≥ 0.0",
        "multiple bboxes in an image, many inferences",
    ): TestCaseMetricsSingleClass(
        Objects=4,
        Inferences=8,
        TP=4,
        FN=0,
        FP=4,
        Precision=0.5,
        Recall=1.0,
        F1=0.6666666666666666,
        AP=0.0,
    ),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.0",
        "multiple bboxes in an image, many inferences",
    ): TestCaseMetricsSingleClass(
        Objects=4,
        Inferences=8,
        TP=4,
        FN=0,
        FP=4,
        Precision=0.5,
        Recall=1.0,
        F1=0.6666666666666666,
        AP=0.0,
    ),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.3",
        "multiple bboxes in an image, many inferences",
    ): TestCaseMetricsSingleClass(
        Objects=4,
        Inferences=8,
        TP=4,
        FN=0,
        FP=4,
        Precision=0.5,
        Recall=1.0,
        F1=0.6666666666666666,
        AP=0.0,
    ),
    (
        "Threshold: F1-Optimal, IoU: 0.5, confidence ≥ 0.1",
        "multiple bboxes in an image, many inferences",
    ): TestCaseMetricsSingleClass(
        Objects=4,
        Inferences=8,
        TP=4,
        FN=0,
        FP=4,
        Precision=0.5,
        Recall=1.0,
        F1=0.6666666666666666,
        AP=0.0,
    ),
    (
        "Threshold: Fixed(0.3), IoU: 0.3, confidence ≥ 0.0",
        "multiple bboxes in an image, too few inferences",
    ): TestCaseMetricsSingleClass(
        Objects=4,
        Inferences=2,
        TP=2,
        FN=2,
        FP=0,
        Precision=1.0,
        Recall=0.5,
        F1=0.6666666666666666,
        AP=0.0,
    ),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.0",
        "multiple bboxes in an image, too few inferences",
    ): TestCaseMetricsSingleClass(
        Objects=4,
        Inferences=2,
        TP=2,
        FN=2,
        FP=0,
        Precision=1.0,
        Recall=0.5,
        F1=0.6666666666666666,
        AP=0.0,
    ),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.3",
        "multiple bboxes in an image, too few inferences",
    ): TestCaseMetricsSingleClass(
        Objects=4,
        Inferences=2,
        TP=2,
        FN=2,
        FP=0,
        Precision=1.0,
        Recall=0.5,
        F1=0.6666666666666666,
        AP=0.0,
    ),
    (
        "Threshold: F1-Optimal, IoU: 0.5, confidence ≥ 0.1",
        "multiple bboxes in an image, too few inferences",
    ): TestCaseMetricsSingleClass(
        Objects=4,
        Inferences=2,
        TP=2,
        FN=2,
        FP=0,
        Precision=1.0,
        Recall=0.5,
        F1=0.6666666666666666,
        AP=0.0,
    ),
    (
        "Threshold: Fixed(0.3), IoU: 0.3, confidence ≥ 0.0",
        "multiple bboxes in an image, suboptimal infs",
    ): TestCaseMetricsSingleClass(
        Objects=4,
        Inferences=2,
        TP=1,
        FN=3,
        FP=1,
        Precision=0.5,
        Recall=0.25,
        F1=0.3333333333333333,
        AP=0.3333333333333333,
    ),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.0",
        "multiple bboxes in an image, suboptimal infs",
    ): TestCaseMetricsSingleClass(
        Objects=4,
        Inferences=2,
        TP=1,
        FN=3,
        FP=1,
        Precision=0.5,
        Recall=0.25,
        F1=0.3333333333333333,
        AP=0.3333333333333333,
    ),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.3",
        "multiple bboxes in an image, suboptimal infs",
    ): TestCaseMetricsSingleClass(
        Objects=4,
        Inferences=2,
        TP=1,
        FN=3,
        FP=1,
        Precision=0.5,
        Recall=0.25,
        F1=0.3333333333333333,
        AP=0.0,
    ),
    (
        "Threshold: F1-Optimal, IoU: 0.5, confidence ≥ 0.1",
        "multiple bboxes in an image, suboptimal infs",
    ): TestCaseMetricsSingleClass(
        Objects=4,
        Inferences=2,
        TP=1,
        FN=3,
        FP=1,
        Precision=0.5,
        Recall=0.25,
        F1=0.3333333333333333,
        AP=0.0,
    ),
    (
        "Threshold: Fixed(0.3), IoU: 0.3, confidence ≥ 0.0",
        "multiple bboxes in an image, ignored matches",
    ): TestCaseMetricsSingleClass(
        Objects=2,
        Inferences=2,
        TP=2,
        FN=0,
        FP=0,
        Precision=1.0,
        Recall=1.0,
        F1=1.0,
        AP=0.0,
    ),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.0",
        "multiple bboxes in an image, ignored matches",
    ): TestCaseMetricsSingleClass(
        Objects=2,
        Inferences=2,
        TP=2,
        FN=0,
        FP=0,
        Precision=1.0,
        Recall=1.0,
        F1=1.0,
        AP=0.0,
    ),
    (
        "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.3",
        "multiple bboxes in an image, ignored matches",
    ): TestCaseMetricsSingleClass(
        Objects=2,
        Inferences=2,
        TP=2,
        FN=0,
        FP=0,
        Precision=1.0,
        Recall=1.0,
        F1=1.0,
        AP=0.0,
    ),
    (
        "Threshold: F1-Optimal, IoU: 0.5, confidence ≥ 0.1",
        "multiple bboxes in an image, ignored matches",
    ): TestCaseMetricsSingleClass(
        Objects=2,
        Inferences=2,
        TP=2,
        FN=0,
        FP=0,
        Precision=1.0,
        Recall=1.0,
        F1=1.0,
        AP=0.0,
    ),
}


@pytest.mark.metrics
@pytest.mark.parametrize(
    "config_name, test_name",
    TEST_PARAMS,
)
def test__object_detection__single_class__compute_test_case_metrics(
    config_name: str,
    test_name: str,
) -> None:
    config = TEST_CONFIGURATIONS[config_name]
    eval = ObjectDetectionEvaluator(configurations=[config])

    random_test_case_name = with_test_prefix("test_evaluator_single_class") + str(uuid.uuid4())
    test_case = TestCase(random_test_case_name, reset=True)

    eval.compute_test_sample_metrics(
        test_case=test_case,
        inferences=TEST_DATA[test_name],
        configuration=config,
    )

    result = eval.compute_test_case_metrics(
        test_case=test_case,
        inferences=TEST_DATA[test_name],
        metrics=[metric for _, metric in EXPECTED_COMPUTE_TEST_SAMPLE_METRICS[config_name][test_name]],
        configuration=config,
    )

    expected = EXPECTED_COMPUTE_TEST_CASE_METRICS[config_name, test_name]
    assert expected == result


@pytest.mark.metrics
@pytest.mark.parametrize(
    "config_name, expected",
    [
        (
            "Threshold: Fixed(0.3), IoU: 0.3, confidence ≥ 0.0",
            TestCaseMetricsSingleClass(
                Objects=35,
                Inferences=32,
                TP=22,
                FN=13,
                FP=10,
                Precision=0.6875,
                Recall=0.6285714285714286,
                F1=0.6567164179104478,
                AP=0.503874883286648,
            ),
        ),
        (
            "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.0",
            TestCaseMetricsSingleClass(
                Objects=35,
                Inferences=31,
                TP=20,
                FN=15,
                FP=11,
                Precision=0.6451612903225806,
                Recall=0.5714285714285714,
                F1=0.606060606060606,
                AP=0.46358543417366943,
            ),
        ),
        (
            "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.3",
            TestCaseMetricsSingleClass(
                Objects=35,
                Inferences=31,
                TP=20,
                FN=15,
                FP=11,
                Precision=0.6451612903225806,
                Recall=0.5714285714285714,
                F1=0.606060606060606,
                AP=0.39375,
            ),
        ),
        (
            "Threshold: F1-Optimal, IoU: 0.5, confidence ≥ 0.1",
            TestCaseMetricsSingleClass(
                Objects=35,
                Inferences=32,
                TP=21,
                FN=14,
                FP=11,
                Precision=0.65625,
                Recall=0.6,
                F1=0.626865671641791,
                AP=0.39375,
            ),
        ),
    ],
)
def test__object_detection__single_class__compute_test_case_metrics__all(
    config_name: str,
    expected: TestCaseMetricsSingleClass,
) -> None:
    config = TEST_CONFIGURATIONS[config_name]
    eval = ObjectDetectionEvaluator(configurations=[config])
    random_test_case_name = with_test_prefix("test_evaluator_single_class") + str(uuid.uuid4())
    test_case = TestCase(random_test_case_name, reset=True)
    eval.compute_test_sample_metrics(
        test_case=test_case,
        inferences=[ts_gt_inf for _, data in TEST_DATA.items() for ts_gt_inf in data],
        configuration=config,
    )
    result = eval.compute_test_case_metrics(
        test_case=test_case,
        inferences=[ts_gt_inf for _, data in TEST_DATA.items() for ts_gt_inf in data],
        metrics=[
            metric for _, metrics in EXPECTED_COMPUTE_TEST_SAMPLE_METRICS[config_name].items() for _, metric in metrics
        ],
        configuration=config,
    )
    assert expected == result
