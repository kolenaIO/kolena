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
from typing import Optional

from pydantic.dataclasses import dataclass

from kolena.workflow import define_workflow
from kolena.workflow import EvaluatorConfiguration
from kolena.workflow import GroundTruth as BaseGroundTruth
from kolena.workflow import Image
from kolena.workflow import ImagePair
from kolena.workflow import Inference as BaseInference
from kolena.workflow import Metadata
from kolena.workflow import MetricsTestCase
from kolena.workflow import MetricsTestSample
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import Keypoints


@dataclass(frozen=True)
class ImageWithMetadata(Image):
    """
    An image belonging to an Image Pair containing the locator and metadata
    for the Face Recognition 1:1 workflow.
    """

    metadata: Metadata = dataclasses.field(default_factory=dict)
    """The metadata associated with an image in the image pair."""


@dataclass(frozen=True)
class TestSample(ImagePair):
    """Test sample type for Face Recognition 1:1 workflow."""

    a: ImageWithMetadata
    """The locator and metadata associated with image A in the test sample."""

    b: ImageWithMetadata
    """The locator and metadata associated with image B in the test sample."""


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    """Ground truth type for Face Recognition 1:1 workflow."""

    is_same: bool
    """Whether to treat this image pair as a a genuine pair (True) or an imposter pair (False)."""


@dataclass(frozen=True)
class Inference(BaseInference):
    """Inference type for Face Recognition 1:1 workflow."""

    a_bbox: Optional[BoundingBox] = None
    """The bounding box associated with image A to be used for face recognition."""

    a_keypoints: Optional[Keypoints] = None
    """The keypoints associated with image A to be used for face recognition."""

    b_bbox: Optional[BoundingBox] = None
    """The bounding box associated with image B to be used for face recognition."""

    b_keypoints: Optional[Keypoints] = None
    """The keypoints associated with image B to be used for face recognition."""

    similarity: float = None
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
class TestSampleMetrics(MetricsTestSample):  # TODO: Include failure to enroll?
    """
    Image-pair-level metrics for Face Recognition 1:1 workflow.
    A test sample is can only be true for one of the following: match, false match (FM), or false non-match (FNM).
    If all categories are false then the sample is a true non-match.
    """

    is_match: bool
    """
    An indication of whether the model correct classified a genuine pair as a genuine pair.
    """

    is_false_match: bool
    """An indication of whether the model incorrectly classified an imposter pair as a genuine pair."""

    is_false_non_match: bool
    """An indication of whether the model incorrectly classified an genuine pair as a imposter pair."""

    failure_to_enroll: bool
    """An indication of whether the model failed to infer."""


@dataclass(frozen=True)
class TestCaseMetrics(MetricsTestCase):
    """Test-case-level aggregate metrics for Face Recognition 1:1 workflow."""

    n_images: int
    """Total number of unique images within this test case."""

    n_genuine_pairs: int
    """Total number of genuine pairs within this test case."""

    n_imposter_pairs: int
    """Total number of imposter pairs within this test case."""

    n_fm: int
    """Total number of false matches within this test case."""

    fmr: float
    """
    The percentage of imposter pairs that are incorrectly classified as genuine pairs
    (i.e. similarity is above threshold) within this test case.
    """

    n_fnm: int
    """Total number of false non-matches within this test case."""

    fnmr: float
    """
    The percentage of genuine pairs that are incorrectly classified as imposter pairs
    (i.e. similarity is below threshold) within this test case.
    """


@dataclass(frozen=True)
class FMRConfiguration(EvaluatorConfiguration):
    """
    False Match Rate (FMR) configuration for Face Recognition 1:1 workflow.
    """

    false_match_rate: Optional[float] = None
    """
    Specify a minimum FMR to apply for predictions.
    """

    def display_name(self) -> str:
        return f"False Match Rate: {self.false_match_rate}"
