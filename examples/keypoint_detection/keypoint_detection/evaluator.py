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
from keypoint_detection.utils import calculate_mse_nmse
from keypoint_detection.utils import compute_distances
from keypoint_detection.workflow import GroundTruth
from keypoint_detection.workflow import Inference
from keypoint_detection.workflow import NmseThreshold
from keypoint_detection.workflow import TestCase
from keypoint_detection.workflow import TestCaseMetrics
from keypoint_detection.workflow import TestSample
from keypoint_detection.workflow import TestSampleMetrics
from keypoint_detection.workflow import TestSuite
from keypoint_detection.workflow import TestSuiteMetrics

from kolena.workflow import AxisConfig
from kolena.workflow import Curve
from kolena.workflow import CurvePlot
from kolena.workflow import Evaluator
from kolena.workflow import Plot


class KeypointsEvaluator(Evaluator):
    plot_by_test_case_name: Dict[str, Plot] = {}

    @staticmethod
    def compute_test_sample_metrics_single(
        test_sample: TestSample,
        ground_truth: GroundTruth,
        inference: Inference,
        configuration: NmseThreshold,
    ) -> TestSampleMetrics:
        if inference.face is None:
            return TestSampleMetrics(match_type="failure_to_detect")

        normalization_factor = ground_truth.normalization_factor
        Δ_nose, norm_Δ_nose = compute_distances(
            ground_truth.face.points[0],
            inference.face.points[0],
            normalization_factor,
        )
        Δ_left_ear, norm_Δ_left_ear = compute_distances(
            ground_truth.face.points[1],
            inference.face.points[1],
            normalization_factor,
        )
        Δ_right_ear, norm_Δ_right_ear = compute_distances(
            ground_truth.face.points[2],
            inference.face.points[2],
            normalization_factor,
        )
        Δ_left_mouth, norm_Δ_left_mouth = compute_distances(
            ground_truth.face.points[3],
            inference.face.points[3],
            normalization_factor,
        )
        Δ_right_mouth, norm_Δ_right_mouth = compute_distances(
            ground_truth.face.points[4],
            inference.face.points[4],
            normalization_factor,
        )
        distances = np.array([Δ_left_ear, Δ_right_ear, Δ_nose, Δ_left_mouth, Δ_right_mouth])
        mse, nmse = calculate_mse_nmse(distances, normalization_factor)
        return TestSampleMetrics(
            match_type="failure_to_align" if nmse > configuration.threshold else "success",
            Δ_left_ear=Δ_left_ear,
            Δ_right_ear=Δ_right_ear,
            Δ_nose=Δ_nose,
            Δ_left_mouth=Δ_left_mouth,
            Δ_right_mouth=Δ_right_mouth,
            normalization_factor=normalization_factor,
            norm_Δ_left_ear=norm_Δ_left_ear,
            norm_Δ_right_ear=norm_Δ_right_ear,
            norm_Δ_nose=norm_Δ_nose,
            norm_Δ_left_mouth=norm_Δ_left_mouth,
            norm_Δ_right_mouth=norm_Δ_right_mouth,
            mse=mse,
            nmse=nmse,
        )

    def compute_test_sample_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        configuration: Optional[NmseThreshold] = None,
    ) -> List[Tuple[TestSample, TestSampleMetrics]]:
        if configuration is None:  # TODO(gh): this is annoying for users to have to deal with
            raise ValueError(f"{type(self).__name__} must have configuration")
        return [
            (
                test_sample,
                self.compute_test_sample_metrics_single(test_sample, ground_truth, inference, configuration),
            )
            for test_sample, ground_truth, inference in inferences
        ]

    def compute_test_case_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        metrics: List[TestSampleMetrics],
        configuration: Optional[NmseThreshold] = None,
    ) -> TestCaseMetrics:
        n_fail_to_align = sum(1 for mts in metrics if mts.match_type == "failure_to_align")
        n_fail_to_detect = sum(1 for mts in metrics if mts.match_type == "failure_to_detect")
        n_fail_total = n_fail_to_align + n_fail_to_detect

        return TestCaseMetrics(
            avg_Δ_left_ear=np.mean([mts.Δ_left_ear for mts in metrics if mts.Δ_left_ear is not None]),
            avg_Δ_right_ear=np.mean([mts.Δ_right_ear for mts in metrics if mts.Δ_right_ear is not None]),
            avg_Δ_nose=np.mean([mts.Δ_nose for mts in metrics if mts.Δ_nose is not None]),
            avg_Δ_left_mouth=np.mean([mts.Δ_left_mouth for mts in metrics if mts.Δ_left_mouth is not None]),
            avg_Δ_right_mouth=np.mean([mts.Δ_right_mouth for mts in metrics if mts.Δ_right_mouth is not None]),
            avg_norm_Δ_left_ear=np.mean([mts.norm_Δ_left_ear for mts in metrics if mts.norm_Δ_left_ear is not None]),
            avg_norm_Δ_right_ear=np.mean(
                [mts.norm_Δ_right_ear for mts in metrics if mts.norm_Δ_right_ear is not None],
            ),
            avg_norm_Δ_nose=np.mean([mts.norm_Δ_nose for mts in metrics if mts.norm_Δ_nose is not None]),
            avg_norm_Δ_left_mouth=np.mean(
                [mts.norm_Δ_left_mouth for mts in metrics if mts.norm_Δ_left_mouth is not None],
            ),
            avg_norm_Δ_right_mouth=np.mean(
                [mts.norm_Δ_right_mouth for mts in metrics if mts.norm_Δ_right_mouth is not None],
            ),
            n_fail_to_align=n_fail_to_align,
            n_fail_to_detect=n_fail_to_detect,
            n_fail_total=n_fail_total,
            total_average_MSE=np.mean([mts.mse for mts in metrics if mts.mse is not None]),
            total_average_NMSE=np.mean([mts.nmse for mts in metrics if mts.nmse is not None]),
            total_detection_failure_rate=n_fail_to_detect / len(metrics) if len(metrics) > 0 else 0,
            total_alignment_failure_rate=n_fail_to_align / len(metrics) if len(metrics) > 0 else 0,
            total_failure_rate=n_fail_total / len(metrics) if len(metrics) > 0 else 0,
        )

    @staticmethod
    def compute_test_case_plot(metrics: List[TestSampleMetrics]) -> Plot:
        def compute_failure_rate(nmse_threshold: float, nmses: List[Optional[float]]) -> float:
            total = sum(1 for nm in nmses if nm is not None and nm > nmse_threshold)
            return total / len(nmses) if len(nmses) > 0 else 0

        nmses = [mts.nmse for mts in metrics]
        x = np.linspace(0, 0.5, 251).tolist()
        y = [compute_failure_rate(x_value, nmses) for x_value in x]
        return CurvePlot(
            title="Alignment Failure Rate vs. NMSE Threshold",
            x_label="NMSE Threshold",
            x_config=AxisConfig(type="log"),
            y_label="Alignment Failure Rate",
            curves=[Curve(label="NMSE", x=x, y=y)],
        )

    def compute_test_case_plots(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        metrics: List[TestSampleMetrics],
        configuration: Optional[NmseThreshold] = None,
    ) -> Optional[List[Plot]]:
        if test_case.name not in self.plot_by_test_case_name.keys():
            self.plot_by_test_case_name[test_case.name] = self.compute_test_case_plot(metrics)
        return [self.plot_by_test_case_name[test_case.name]]

    def compute_test_suite_metrics(
        self,
        test_suite: TestSuite,
        metrics: List[Tuple[TestCase, TestCaseMetrics]],
        configuration: Optional[NmseThreshold] = None,
    ) -> Optional[TestSuiteMetrics]:
        return TestSuiteMetrics(
            variance_average_MSE=np.var([m.total_average_MSE for _, m in metrics]),
            variance_average_NMSE=np.var([m.total_average_NMSE for _, m in metrics]),
            variance_detection_failure_rate=np.var([m.total_detection_failure_rate for _, m in metrics]),
            variance_alignment_failure_rate=np.var([m.total_alignment_failure_rate for _, m in metrics]),
            variance_failure_rate=np.var([m.total_failure_rate for _, m in metrics]),
        )
