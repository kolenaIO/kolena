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
import numpy as np
import pytest
from face_recognition_11.evaluator import FaceRecognition11Evaluator
from face_recognition_11.workflow import Inference

# TODO: create better test case

MOCK_SCORES = np.array([0.1, 0.11, 0.23, 0.37, 0.44, 0.67, 0.67, 0.68, 0.70, 0.71])
MOCK_MATCH_1 = np.array(
    [0.16723002, 0.25111065, 0.28177444, 0.42642674, 0.59722537, 0.74354211, 0.78708682, 0.81465805, 0.87341382],
)
MOCK_MATCH_2 = np.array([0.59722537, 0.74354211, 0.78708682, 0.81465805, 0.87341382])
MOCK_MATCH_3 = np.array([0.87341382])


@pytest.fixture
def mock_inferences():
    inferences = list([Inference(similarity=s) for s in MOCK_SCORES])
    return inferences


@pytest.mark.parametrize(
    "fmr, expected",
    [(0.1, MOCK_MATCH_1), (0.5, MOCK_MATCH_2), (0.90, MOCK_MATCH_3)],
)
def test_compute_threshold(fmr, expected, mock_inferences):
    threshold = FaceRecognition11Evaluator.compute_threshold(mock_inferences, fmr)
    genuine_pairs = np.array([inf.similarity for inf in mock_inferences if inf.similarity > threshold])
    np.testing.assert_array_equal(genuine_pairs, expected)
