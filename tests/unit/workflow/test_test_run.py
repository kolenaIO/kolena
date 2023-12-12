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
            },
        ),
    ]

    # Define the expected output
    expected_updated_records = [
        (
            {"locator": "s3://bucket/image1.jpg", "data_type": "TEST_SAMPLE/IMAGE"},
            {
                "metric_1": "value1",
            },
        ),
    ]
    expected_removed_items = []

    # Call the method under test
    updated_records, removed_items = test_run._extract_thresholded_metrics(records)
    # Assert the results
    assert updated_records == expected_updated_records
    assert removed_items == expected_removed_items


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
    expected_updated_records = [
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
    expected_removed_items = [
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
    updated_records, removed_items = test_run._extract_thresholded_metrics(records)
    # Assert the results with descriptive error messages
    assert updated_records == expected_updated_records, "Updated records do not match expected values"
    assert removed_items == expected_removed_items, "Removed items do not match expected values"

    # Optionally, compare lists element by element for more detailed error reporting
    for i in range(len(expected_updated_records)):
        assert updated_records[i] == expected_updated_records[i], f"Record at index {i} does not match"

    for i in range(len(expected_removed_items)):
        assert removed_items[i] == expected_removed_items[i], f"Removed item at index {i} does not match"
