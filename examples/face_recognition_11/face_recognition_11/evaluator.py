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
from typing import Optional
from typing import Tuple

import numpy as np
from face_recognition_11.workflow import GroundTruth
from face_recognition_11.workflow import Inference
from face_recognition_11.workflow import FMRThresholdConfiguration
from face_recognition_11.workflow import TestCase
from face_recognition_11.workflow import TestCaseMetrics
from face_recognition_11.workflow import TestSample
from face_recognition_11.workflow import TestSampleMetrics
from face_recognition_11.workflow import TestSuite
from face_recognition_11.workflow import TestSuiteMetrics

from kolena.workflow import AxisConfig
from kolena.workflow import Curve
from kolena.workflow import CurvePlot
from kolena.workflow import Evaluator
from kolena.workflow import Plot


class FaceRecognition11(Evaluator):
    @staticmethod
    def compute_test_sample_metrics_single(
        test_sample: TestSample,
        ground_truth: GroundTruth,
        inference: Inference,
        configuration: FMRThresholdConfiguration,
    ) -> TestSampleMetrics:
        if inference.similarity is None:
            return TestCaseMetrics(ignore=True)

        return TestSampleMetrics(
            ignore=True,
            # TODO: FM,FNM
            is_false_match=None,
            is_false_non_match=None,
            is_match=ground_truth.is_same,
            threshold=configuration.threshold,
        )

    def compute_test_sample_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        configuration: Optional[FMRThresholdConfiguration] = None,
    ) -> List[Tuple[TestSample, TestSampleMetrics]]:
        if configuration is None:  # TODO(gh): this is annoying for users to have to deal with
            raise ValueError(f"{type(self).__name__} must have configuration")
        return [
            (test_sample, self.compute_test_sample_metrics_single(test_sample, ground_truth, inference, configuration))
            for test_sample, ground_truth, inference in inferences
        ]

    def compute_test_case_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        metrics: List[TestSampleMetrics],
        configuration: Optional[FMRThresholdConfiguration] = None,
    ) -> TestCaseMetrics:
        return TestCaseMetrics

    def compute_test_case_plots(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        metrics: List[TestSampleMetrics],
        configuration: Optional[FMRThresholdConfiguration] = None,
    ) -> Optional[List[Plot]]:
        plots = []
        # ROC curve

        # Test Case FNMR vs Baseline FMR
        return plots

    def compute_test_suite_metrics(
        self,
        test_suite: TestSuite,
        metrics: List[Tuple[TestCase, TestCaseMetrics]],
        configuration: Optional[FMRThresholdConfiguration] = None,
    ) -> Optional[TestSuiteMetrics]:
        return TestSuiteMetrics
