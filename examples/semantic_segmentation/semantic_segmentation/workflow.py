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
from enum import Enum
from typing import Dict

from pydantic.dataclasses import dataclass

from kolena.workflow import define_workflow
from kolena.workflow import EvaluatorConfiguration
from kolena.workflow import GroundTruth as BaseGroundTruth
from kolena.workflow import Image
from kolena.workflow import Inference as BaseInference
from kolena.workflow import Metadata
from kolena.workflow import MetricsTestCase
from kolena.workflow import MetricsTestSample
from kolena.workflow.annotation import BitmapMask
from kolena.workflow.annotation import SegmentationMask
from kolena.workflow.asset import BinaryAsset


@dataclass(frozen=True)
class TestSample(Image):
    metadata: Metadata = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    """Ground truth type for the Semantic Segmentation workflow."""

    mask: SegmentationMask
    """
    The ground truth [`SegmentationMask`][kolena.workflow.annotation.SegmentationMask] associated with an image.
    """


@dataclass(frozen=True)
class Inference(BaseInference):
    """Inference type for the Semantic Segmentation workflow."""

    prob: BinaryAsset
    """
    The numpy array of probabilities produced by model stored in `.npy` file.
    """

    activation_map: BitmapMask
    """
    The png encoded bitmap mask that corresponds to an activation map based on the probability array.
    """


_, TestCase, TestSuite, Model = define_workflow("Semantic Segmentation", TestSample, GroundTruth, Inference)


@dataclass(frozen=True)
class TestSampleMetric(MetricsTestSample):
    """Sample-level metrics for the Semantic Segmentation workflow."""

    TP: SegmentationMask
    FP: SegmentationMask
    FN: SegmentationMask

    Precision: float
    Recall: float
    F1: float

    CountTP: int
    CountFP: int
    CountFN: int


@dataclass(frozen=True)
class TestCaseMetric(MetricsTestCase):
    """Test-case-level aggregate metrics for the Semantic Segmentation workflow."""

    Precision: float
    Recall: float
    F1: float
    AP: float


class Label(Enum):
    PERSON = 1

    @classmethod
    def as_label_map(cls) -> Dict[int, str]:
        return {option.value: option.name for option in cls}


class SegmentationConfiguration(EvaluatorConfiguration):
    threshold: float

    def display_name(self) -> str:
        return f"T={self.threshold}"
