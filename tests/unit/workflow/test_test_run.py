# Copyright 2021-2024 Kolena Inc.
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
from unittest.mock import patch

import pytest

from kolena.workflow.test_run import TestRun


@pytest.fixture
def mock_test_run_init():
    with patch.object(TestRun, "__init__", lambda *args, **kwargs: None):
        yield


@pytest.fixture
def mock_krequests_put():
    with patch("kolena.workflow.test_run.krequests.put") as mock_put:
        yield mock_put


def test__when_no_threshold_metric_classes(mock_test_run_init, mock_krequests_put) -> None:
    # Create a TestRun instance
    test_run = TestRun()

    # Define the input records
    records = [
        (
            {"locator": "s3://bucket/image1.jpg", "data_type": "TEST_SAMPLE/IMAGE"},
            {
                "metric_1": "value1",
                "metric_2": False,
                "metric_3": 12.0,
            },
        ),
    ]

    # Define the expected output
    expected_standard_metrics = [
        (
            {"locator": "s3://bucket/image1.jpg", "data_type": "TEST_SAMPLE/IMAGE"},
            {
                "metric_1": "value1",
                "metric_2": False,
                "metric_3": 12.0,
            },
        ),
    ]
    expected_thresholded_metrics = []

    # Call the method under test
    standard_metrics, thresholded_metrics = test_run._extract_thresholded_metrics(records)
    # Assert the results
    assert standard_metrics == expected_standard_metrics
    assert thresholded_metrics == expected_thresholded_metrics


def test__when_no_threshold_metric_classes_non_dict_item(mock_test_run_init, mock_krequests_put) -> None:
    # Create a TestRun instance
    test_run = TestRun()

    # Define the input records with a non-dict item in the list
    records = [
        (
            {"locator": "s3://bucket/image1.jpg", "data_type": "TEST_SAMPLE/IMAGE"},
            {
                "metric_1": ["non-dict-item", 1, 12.0, True, {"data_type": "METRICS/REGULAR"}],
            },
        ),
    ]

    # Define the expected output
    expected_standard_metrics = [
        (
            {"locator": "s3://bucket/image1.jpg", "data_type": "TEST_SAMPLE/IMAGE"},
            {
                "metric_1": ["non-dict-item", 1, 12.0, True, {"data_type": "METRICS/REGULAR"}],
            },
        ),
    ]
    expected_thresholded_metrics = []

    # Call the method under test
    standard_metrics, thresholded_metrics = test_run._extract_thresholded_metrics(records)

    # Assert the results
    assert standard_metrics == expected_standard_metrics
    assert thresholded_metrics == expected_thresholded_metrics


def test__extract_thresholded_metrics(mock_test_run_init, mock_krequests_put) -> None:
    # Create a TestRun instance
    test_run = TestRun()

    # Define the input records
    records = [
        (
            {"locator": "s3://bucket/image2.jpg", "data_type": "TEST_SAMPLE/IMAGE"},
            {
                "metric_1": "value1",
                "class_1": [
                    {"threshold": "0.1", "data_type": "METRICS/THRESHOLDED", "value": 10},
                    {"threshold": "0.2", "data_type": "METRICS/THRESHOLDED", "value": 20},
                ],
            },
        ),
        (
            {"locator": "s3://bucket/image2.jpg", "data_type": "TEST_SAMPLE/IMAGE"},
            {
                "metric_1": "value1",
                "class_1": [
                    {"threshold": "0.1", "data_type": "METRICS/THRESHOLDED", "value": 10},
                    {"threshold": "0.2", "data_type": "METRICS/THRESHOLDED", "value": 20},
                ],
                "class_2": [
                    {"threshold": "0.1", "data_type": "METRICS/THRESHOLDED", "value": 10},
                    {"threshold": "0.2", "data_type": "METRICS/THRESHOLDED", "value": 20},
                ],
            },
        ),
    ]

    # Define the expected output
    expected_standard_metrics = [
        (
            {"locator": "s3://bucket/image2.jpg", "data_type": "TEST_SAMPLE/IMAGE"},
            {
                "metric_1": "value1",
            },
        ),
        (
            {"locator": "s3://bucket/image2.jpg", "data_type": "TEST_SAMPLE/IMAGE"},
            {
                "metric_1": "value1",
            },
        ),
    ]
    expected_thresholded_metrics = [
        (
            {"locator": "s3://bucket/image2.jpg", "data_type": "TEST_SAMPLE/IMAGE"},
            {"name": "class_1", "threshold": "0.1", "data_type": "METRICS/THRESHOLDED", "value": 10},
        ),
        (
            {"locator": "s3://bucket/image2.jpg", "data_type": "TEST_SAMPLE/IMAGE"},
            {"name": "class_1", "threshold": "0.2", "data_type": "METRICS/THRESHOLDED", "value": 20},
        ),
        (
            {"locator": "s3://bucket/image2.jpg", "data_type": "TEST_SAMPLE/IMAGE"},
            {"name": "class_1", "threshold": "0.1", "data_type": "METRICS/THRESHOLDED", "value": 10},
        ),
        (
            {"locator": "s3://bucket/image2.jpg", "data_type": "TEST_SAMPLE/IMAGE"},
            {"name": "class_1", "threshold": "0.2", "data_type": "METRICS/THRESHOLDED", "value": 20},
        ),
        (
            {"locator": "s3://bucket/image2.jpg", "data_type": "TEST_SAMPLE/IMAGE"},
            {"name": "class_2", "threshold": "0.1", "data_type": "METRICS/THRESHOLDED", "value": 10},
        ),
        (
            {"locator": "s3://bucket/image2.jpg", "data_type": "TEST_SAMPLE/IMAGE"},
            {"name": "class_2", "threshold": "0.2", "data_type": "METRICS/THRESHOLDED", "value": 20},
        ),
    ]

    # Call the method under test
    standard_metrics, thresholded_metrics = test_run._extract_thresholded_metrics(records)
    # Assert the results with descriptive error messages
    assert standard_metrics == expected_standard_metrics, "Updated records do not match expected values"
    assert thresholded_metrics == expected_thresholded_metrics, "Removed items do not match expected values"

    # Optionally, compare lists element by element for more detailed error reporting
    for i in range(len(expected_standard_metrics)):
        assert standard_metrics[i] == expected_standard_metrics[i], f"Record at index {i} does not match"

    for i in range(len(expected_thresholded_metrics)):
        assert thresholded_metrics[i] == expected_thresholded_metrics[i], f"Removed item at index {i} does not match"
