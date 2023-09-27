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
import pytest
import numpy as np

from face_recognition_11.evaluator import FaceRecognition11Evaluator


@pytest.mark.parametrize(
    "data, threshold, expected",
    [
        (np.array([1, 2, 3, 4, 5]), 0.5, np.array([0, 0, 0, 4, 5])),
        (np.array([1, 2, 3, 4, 5]), 0.2, np.array([0, 0, 0, 0, 5])),
        (np.array([10, 20, 30, 40, 50]), 0.8, np.array([0, 0, 0, 0, 50])),
    ],
)
def test_compute_threshold(data, threshold, expected):
    result = FaceRecognition11Evaluator.compute_threshold(data, threshold)
    np.testing.assert_array_equal(result, expected)
