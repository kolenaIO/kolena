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
from face_recognition_11.workflow import FMRConfiguration
from face_recognition_11.workflow import TestCase
from face_recognition_11.workflow import TestCaseMetrics
from face_recognition_11.workflow import TestSample
from face_recognition_11.workflow import TestSampleMetrics
from face_recognition_11.workflow import TestSuite

from kolena.workflow import AxisConfig
from kolena.workflow import Curve
from kolena.workflow import CurvePlot
from kolena.workflow import Evaluator
from kolena.workflow import Plot
from kolena.workflow import TestCases
from kolena.workflow import EvaluationResults


class FaceRecognition11Evaluator(Evaluator):
    @staticmethod
    def compute_threshold(inferences: List[Inference], fmr: float) -> float:
        """
        The threhsold is calculated as Threshold = Quantile_v(1 - FMR) where the quantiles are of the observed
        IMPOSTER scores, v.

        For more details on how threshold is used see https://pages.nist.gov/frvt/reports/11/frvt_11_report.pdf
        """
        similarity_scores = np.array([inf.similarity for inf in inferences])
        threshold = np.quantile(similarity_scores, 1 - fmr)
        return threshold

    @staticmethod
    def compute_test_sample_metrics_single(
        ground_truth: GroundTruth,
        inference: Inference,
        threshold: float,
    ) -> TestSampleMetrics:
        is_match, is_false_match, is_false_non_match = False, False, False

        if inference.similarity is None:
            return TestSampleMetrics(
                is_match=is_match,
                is_false_match=is_false_match,
                is_false_non_match=is_false_non_match,
                failure_to_enroll=True,
            )

        if inference.similarity > threshold:  # match
            if ground_truth.is_same:
                is_match = True
            else:
                is_false_match = True
        else:  # no match
            if ground_truth.is_same:
                is_false_non_match = True

        return TestSampleMetrics(
            is_match=is_match,
            is_false_match=is_false_match,
            is_false_non_match=is_false_non_match,
            failure_to_enroll=False,
        )

    def compute_test_sample_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        configuration: Optional[FMRConfiguration] = None,
    ) -> List[Tuple[TestSample, TestSampleMetrics]]:
        inferences_of_imposter_pairs = [
            inf for ts, gt, inf in inferences if (not gt.is_same) and (inf.similarity is not None)
        ]
        threshold = self.compute_threshold(inferences_of_imposter_pairs, configuration.false_match_rate)

        return [
            (
                test_sample,
                self.compute_test_sample_metrics_single(ground_truth, inference, threshold),
            )
            for test_sample, ground_truth, inference in inferences
        ]

    def compute_test_case_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        metrics: List[TestSampleMetrics],
        configuration: Optional[FMRConfiguration] = None,
    ) -> TestCaseMetrics:
        n_genuine_pairs = np.sum([gt.is_same for ts, gt, inf in inferences])
        n_imposter_pairs = np.sum([not gt.is_same for ts, gt, inf in inferences])

        # do not count FTE as belonging to FM or FNM
        n_fm = np.sum([metric.is_false_match and not metric.failure_to_enroll for metric in metrics])
        n_fnm = np.sum([metric.is_false_non_match and not metric.failure_to_enroll for metric in metrics])

        return TestCaseMetrics(
            n_images=len(metrics) * 2,
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
        configuration: Optional[FMRConfiguration] = None,
    ) -> Optional[List[Plot]]:
        inferences_of_imposter_pairs = [
            inf for ts, gt, inf in inferences if (not gt.is_same) and (inf.similarity is not None)
        ]
        FMR_lower = -6
        FMR_upper = -1
        baseline_fmr_x = list(np.logspace(FMR_lower, FMR_upper, 10))

        thresholds = [self.compute_threshold(inferences_of_imposter_pairs, fmr) for fmr in baseline_fmr_x]

        n_genuine_pairs = np.sum([gt.is_same for ts, gt, inf in inferences])
        n_imposter_pairs = np.sum([not gt.is_same for ts, gt, inf in inferences])

        fnmr_y = list()
        fmr_y = list()

        for threshold in thresholds:
            tsm_for_one_threshold = [
                self.compute_test_sample_metrics_single(gt, inf, threshold) for ts, gt, inf in inferences
            ]
            n_fm = np.sum([metric.is_false_match and not metric.failure_to_enroll for metric in tsm_for_one_threshold])
            n_fnm = np.sum(
                [metric.is_false_non_match and not metric.failure_to_enroll for metric in tsm_for_one_threshold]
            )
            fnmr_y.append(n_fnm / n_genuine_pairs)
            fmr_y.append(n_fm / n_imposter_pairs)

        curve_test_case_fnmr = CurvePlot(
            title="Test Case FNMR vs. Baseline FMR",
            x_label="Baseline False Match Rate",
            y_label="Test Case False Non-Match Rate (%)",
            x_config=AxisConfig(type="log"),
            curves=[Curve(x=baseline_fmr_x, y=fnmr_y)],
        )

        curve_test_case_fmr = CurvePlot(
            title="Test Case FMR vs. Baseline FMR",
            x_label="Baseline False Match Rate",
            y_label="Test Case False Match Rate",
            x_config=AxisConfig(type="log"),
            y_config=AxisConfig(type="log"),
            curves=[Curve(x=baseline_fmr_x, y=fmr_y)],
        )

        return [curve_test_case_fnmr, curve_test_case_fmr]
