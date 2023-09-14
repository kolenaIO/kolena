# build out the workflow for the FR problem which is already supported in kolena.fr as a pre-built workflow.

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
import dataclasses
from typing import List, Tuple
from typing import Optional

from pydantic.dataclasses import dataclass

from kolena.workflow import define_workflow
from kolena.workflow import EvaluatorConfiguration
from kolena.workflow import GroundTruth as BaseGroundTruth
from kolena.workflow import ImagePair
from kolena.workflow import Inference as BaseInference
from kolena.workflow import Metadata
from kolena.workflow import MetricsTestCase
from kolena.workflow import MetricsTestSample
from kolena.workflow import MetricsTestSuite
from kolena.workflow.annotation import BoundingBox, Keypoints


@dataclass(frozen=True)
class TestSample(ImagePair):
    """Test sample type for Face Recognition 1:1 workflow."""

    left: Metadata = dataclasses.field(default_factory=dict)
    """The metadata associated with the left image in the test sample."""

    right: Metadata = dataclasses.field(default_factory=dict)
    """The metadata associated with the right image in the test sample."""


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    """Ground truth type for Face Recognition 1:1 workflow."""

    is_same: bool
    """Whether to treat this image pair as a a genuine pair (True) or an imposter pair (False)."""


@dataclass(frozen=True)
class Inference(BaseInference):
    """Inference type for Face Recognition 1:1 workflow."""

    left: Tuple[BoundingBox, Keypoints]
    """The bounding box and keypoints associated with the left image to be used for face recognition."""

    right: Tuple[BoundingBox, Keypoints]
    """The bounding box and keypoints associated with the right image to be used for face recognition."""

    similarity: Optional[float]
    """
    The similarity score computed between the two embeddings in this image pair. Should be left empty when either
    image in the pair is a failure to enroll.
    """


workflow, TestCase, TestSuite, Model = define_workflow(
    "Face Recognition 1:1",
    TestSample,
    GroundTruth,
    Inference,
)


@dataclass(frozen=True)
class TestSampleMetrics(MetricsTestSample):
    """Image-pair-level metrics for Face Recognition 1:1 workflow."""

    ignore: bool

    is_false_match: Optional[bool]
    """An indication of whether the model incorrectly classified an imposter pair as a genuine pair."""

    is_false_non_match: Optional[bool]
    """An indication of whether the model incorrectly classified an genuine pair as a imposter pair."""

    is_match: Optional[bool]
    """An indication of whether the image pair form a genuine pair (True) or an imposter pair (False)."""

    threshold: Optional[float]
    """
    The threshold used in evaluation - specified by the `FMRThresholdConfiguration`.
    """


@dataclass(frozen=True)
class TestCaseMetrics(MetricsTestCase):
    """Test-case-level aggregate metrics for Face Recognition 1:1 workflow."""

    n_samples: int
    """Total number of samples (pair of source and target images) within this test case."""

    n_images: int
    """Total number of source and target images within this test case."""

    n_tp: int
    """Total number of true positives within in this test case."""

    n_tn: int
    """Total number of true negatives within in this test case."""

    n_fp: int
    """Total number of false positives within in this test case."""

    n_fn: int
    """Total number of false negatives within in this test case."""

    n_imposter_pairs: int
    """Total number of imposter pairs within in this test case."""

    n_ignored_pairs: int
    """Total number of ignored pairs within in this test case."""

    n_genuine_pairs: int
    """Total number of genuine pairs within in this test case."""

    auc: float
    correct_rate: float
    incorrect_rate: float
    fmr: float
    Δ_fmr: float
    fnmr: float
    Δ_fnmr: float
    recall: float


@dataclass(frozen=True)
class TestSuiteMetrics(MetricsTestSuite):
    """Test-suite-level metrics for Face Recognition 1:1 workflow."""


@dataclass(frozen=True)
class FMRThresholdConfiguration(EvaluatorConfiguration):
    """
    False Match Rate (FMR) threshold configuration for Face Recognition 1:1 workflow.
    Specify a minimum
    """

    threshold: Optional[float] = None
    """
    FMR threshold to apply for predictions.
    """
