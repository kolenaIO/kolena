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
        is_match, is_false_match, is_false_non_match = False, False, False

        if inference.similarity is None:
            return TestCaseMetrics(is_match, is_false_match, is_false_non_match)

        if inference.similarity > configuration.threshold:  # match
            is_match = True
        else:  # no match
            is_false_match = not ground_truth.is_same
            is_false_non_match = ground_truth.is_same

        return TestCaseMetrics(is_match, is_false_match, is_false_non_match)

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
        test_case: TestCase,  # NOTE: What is TestCase?
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        metrics: List[TestSampleMetrics],
        configuration: Optional[FMRThresholdConfiguration] = None,
    ) -> TestCaseMetrics:
        n_genuine_pairs = np.sum([gt.is_same for _, gt, _ in inferences])
        n_imposter_pairs = np.sum([not gt.is_same for _, gt, _ in inferences])
        n_fm = np.sum([metric.is_false_match for metric in metrics])
        n_fnm = np.sum([metric.is_false_non_match for metric in metrics])

        return TestCaseMetrics(
            n_images=len(metrics),
            n_genuine_pairs=n_genuine_pairs,
            n_imposter_pairs=n_imposter_pairs,
            n_fm=n_fm,
            fmr=n_fm / n_imposter_pairs,
            n_fnm=n_fnm,
            fnmr=n_fnm / n_genuine_pairs,
        )

    def compute_test_case_plots(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        metrics: List[TestSampleMetrics],
        configuration: Optional[FMRThresholdConfiguration] = None,
    ) -> Optional[List[Plot]]:
        plots = []
        # TODO: existing plots depend on baseline - add in baseline fmr

        # set Test Sample as the first baseline to compute remaining metrics

        # AUC and ROC plots

        return plots
