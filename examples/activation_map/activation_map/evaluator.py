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

from activation_map.workflow import GroundTruth
from activation_map.workflow import Inference
from activation_map.workflow import TestSample

from kolena.workflow import EvaluationResults
from kolena.workflow import MetricsTestCase
from kolena.workflow import MetricsTestSample
from kolena.workflow import TestCases


def evaluate_dummy(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
) -> EvaluationResults:
    metrics_test_sample = [(ts, MetricsTestSample()) for ts in test_samples]
    metrics_test_case = [
        (tc, MetricsTestCase())
        for tc, *_ in test_cases.iter(test_samples, ground_truths, inferences, metrics_test_sample)
    ]
    return EvaluationResults(
        metrics_test_sample=metrics_test_sample,
        metrics_test_case=metrics_test_case,
    )
