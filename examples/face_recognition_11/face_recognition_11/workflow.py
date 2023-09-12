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

    similarity: float
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

# TODO: implement Metric Classes
