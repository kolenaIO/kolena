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
from typing import Tuple

import pytest

from .test_evaluator_single_class_test_sample_metrics import EXPECTED_COMPUTE_TEST_SAMPLE_METRICS
from .test_evaluator_single_class_test_sample_metrics import TEST_CASE
from .test_evaluator_single_class_test_sample_metrics import TEST_CONFIGURATIONS
from .test_evaluator_single_class_test_sample_metrics import TEST_DATA
from .test_evaluator_single_class_test_sample_metrics import TEST_PARAMS
from kolena._experimental.object_detection import ObjectDetectionEvaluator
from kolena._experimental.object_detection.workflow import TestCaseMetricsSingleClass


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
        AP=0.07494252873563219,
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
        AP=0.08547008547008546,
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
        AP=0.0967741935483871,
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
        AP=0.11160714285714286,
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
        AP=0.1264367816091954,
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
        AP=0.1411764705882353,
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
        AP=0.1557603686635945,
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
        AP=0.1701388888888889,
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
        AP=0.225,
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
        AP=0.28329725829725827,
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
        AP=0.3350378787878787,
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
        AP=0.3806818181818181,
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
        AP=0.3942422161172161,
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
        AP=0.3817016317016317,
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
        AP=0.3711588541666667,
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
        AP=0.3621724170437406,
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
        AP=0.3827833560704156,
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
        AP=0.40901272789817683,
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
        AP=0.4223004694835681,
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
        AP=0.4346145596145596,
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
        AP=0.4342707420203166,
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
        AP=0.4345179360674273,
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
        AP=0.4351511437908496,
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
        AP=0.43603602058319035,
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
        AP=0.43606913919413914,
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
        AP=0.4361849489042472,
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
        AP=0.43637018384940185,
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
        AP=0.43661370697345353,
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
        AP=0.43317498999317183,
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
        AP=0.42995794311481095,
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
        AP=0.42000471636742126,
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
        AP=0.41066791911045947,
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
        AP=0.4177783244947424,
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
        AP=0.42473157586898386,
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
        AP=0.4315321386379141,
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
        AP=0.43818436465495286,
    ),
}


@pytest.mark.metrics
@pytest.mark.parametrize(
    "config_name, test_name",
    TEST_PARAMS,
)
def test__prebuilt__object__detection__single__class__compute__test__case__metrics(
    config_name: str,
    test_name: str,
) -> None:
    config = TEST_CONFIGURATIONS[config_name]

    eval = ObjectDetectionEvaluator(configurations=[config])
    eval.compute_test_sample_metrics(
        test_case=TEST_CASE,
        inferences=TEST_DATA[test_name],
        configuration=config,
    )

    result = eval.compute_test_case_metrics(
        test_case=TEST_CASE,
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
                AP=0.4511139574910939,
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
                AP=0.45320146520146526,
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
                AP=0.4446082570959419,
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
                AP=0.43818436465495286,
            ),
        ),
    ],
)
def test__prebuilt__object__detection__single__class__compute__test__case__metrics__all(
    config_name: str,
    expected: TestCaseMetricsSingleClass,
) -> None:
    config = TEST_CONFIGURATIONS[config_name]
    eval = ObjectDetectionEvaluator(configurations=[config])
    eval.compute_test_sample_metrics(
        test_case=TEST_CASE,
        inferences=[ts_gt_inf for _, data in TEST_DATA.items() for ts_gt_inf in data],
        configuration=config,
    )
    result = eval.compute_test_case_metrics(
        test_case=TEST_CASE,
        inferences=[ts_gt_inf for _, data in TEST_DATA.items() for ts_gt_inf in data],
        metrics=[
            metric for _, metrics in EXPECTED_COMPUTE_TEST_SAMPLE_METRICS[config_name].items() for _, metric in metrics
        ],
        configuration=config,
    )

    assert expected == result
