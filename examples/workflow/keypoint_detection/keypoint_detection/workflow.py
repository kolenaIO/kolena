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
from dataclasses import dataclass
from typing import Optional
from typing import Union

import numpy as np

from kolena.workflow import define_workflow
from kolena.workflow import EvaluatorConfiguration
from kolena.workflow import GroundTruth as GT
from kolena.workflow import Image
from kolena.workflow import Inference as Inf
from kolena.workflow import MetricsTestCase
from kolena.workflow import MetricsTestSample
from kolena.workflow import MetricsTestSuite
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import Keypoints


@dataclass(frozen=True)
class TestSample(Image):
    bbox: Optional[BoundingBox] = None


@dataclass(frozen=True)
class GroundTruth(GT):
    face: Keypoints

    # In order to compute norm error, some normalization factor describing
    # the size of the face in the image is required.
    normalization_factor: float


@dataclass(frozen=True)
class Inference(Inf):
    face: Keypoints


# use these TestCase, TestSuite, and Model definitions to create and run tests
wf, TestCase, TestSuite, Model = define_workflow(
    "Keypoint Detection",
    TestSample,
    GroundTruth,
    Inference,
)


@dataclass(frozen=True)
class TestSampleMetrics(MetricsTestSample):
    match_type: str  # one of "failure_to_detect", "failure_to_align", "success"
    # if match_type is "failure_to_detect", metrics below are None
    Δ_nose: Optional[float] = None
    Δ_left_eye: Optional[float] = None
    Δ_right_eye: Optional[float] = None
    Δ_left_mouth: Optional[float] = None
    Δ_right_mouth: Optional[float] = None
    normalization_factor: Optional[float] = None
    norm_Δ_nose: Optional[float] = None
    norm_Δ_left_eye: Optional[float] = None
    norm_Δ_right_eye: Optional[float] = None
    norm_Δ_left_mouth: Optional[float] = None
    norm_Δ_right_mouth: Optional[float] = None
    mse: Optional[float] = None
    nmse: Optional[float] = None


@dataclass(frozen=True)
class TestCaseMetrics(MetricsTestCase):
    avg_Δ_nose: Union[float, np.floating]
    avg_Δ_left_eye: Union[float, np.floating]
    avg_Δ_right_eye: Union[float, np.floating]
    avg_Δ_left_mouth: Union[float, np.floating]
    avg_Δ_right_mouth: Union[float, np.floating]
    avg_norm_Δ_nose: Union[float, np.floating]
    avg_norm_Δ_left_eye: Union[float, np.floating]
    avg_norm_Δ_right_eye: Union[float, np.floating]
    avg_norm_Δ_left_mouth: Union[float, np.floating]
    avg_norm_Δ_right_mouth: Union[float, np.floating]
    n_fail_to_align: int
    n_fail_to_detect: int
    n_fail_total: int
    total_average_MSE: Union[float, np.floating]
    total_average_NMSE: Union[float, np.floating]
    total_detection_failure_rate: float
    total_alignment_failure_rate: float
    total_failure_rate: float


@dataclass(frozen=True)
class TestSuiteMetrics(MetricsTestSuite):
    variance_average_MSE: Union[float, np.floating]
    variance_average_NMSE: Union[float, np.floating]
    variance_detection_failure_rate: Union[float, np.floating]
    variance_alignment_failure_rate: Union[float, np.floating]
    variance_failure_rate: Union[float, np.floating]


@dataclass(frozen=True)
class NmseThreshold(EvaluatorConfiguration):
    # threshold for NMSE (norm mean square error) above which an image is considered an "alignment failure"
    threshold: float

    def display_name(self) -> str:
        return f"nme-threshold-{self.threshold}"
