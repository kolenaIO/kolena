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
import dataclasses
from typing import List
from typing import Union

from kolena._experimental.object_detection.workflow import TestSample as BaseTestSample
from kolena._experimental.object_detection.workflow import ThresholdConfiguration
from kolena._utils.pydantic_v1.dataclasses import dataclass
from kolena.workflow import define_workflow
from kolena.workflow import GroundTruth as BaseGroundTruth
from kolena.workflow import Inference as BaseInference
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import LabeledPolygon
from kolena.workflow.annotation import ScoredLabeledPolygon


@dataclass(frozen=True)
class TestSample(BaseTestSample):
    """The [`Image`][kolena.workflow.Image] sample type for the pre-built 2D Instance Segmentation workflow."""


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    """Ground truth type for the pre-built 2D Instance Segmentation workflow."""

    polygons: List[LabeledPolygon]
    """
    The ground truth [`LabeledPolygons`][kolena.workflow.annotation.LabeledPolygon] associated with an image.
    """

    ignored_polygons: List[Union[LabeledBoundingBox, LabeledPolygon]] = dataclasses.field(default_factory=list)
    """
    The ground truth [`LabeledPolygons`][kolena.workflow.annotation.LabeledPolygon] to be ignored
    in evaluation associated with an image.
    """

    n_polygons: int = dataclasses.field(default_factory=lambda: 0)

    def __post_init__(self):
        object.__setattr__(self, "n_polygons", len(self.polygons))


@dataclass(frozen=True)
class Inference(BaseInference):
    """Inference type for the pre-built 2D Instance Segmentation workflow."""

    polygons: List[ScoredLabeledPolygon]
    """
    The inference [`ScoredLabeledPolygons`][kolena.workflow.annotation.ScoredLabeledPolygon] associated with an image.
    """

    ignored: bool = False
    """
    Whether the image (and its associated inference `polygons`) should be ignored
    in evaluating the results of the model.
    """


_, TestCase, TestSuite, Model = define_workflow(
    "Instance Segmentation",
    TestSample,
    GroundTruth,
    Inference,
)


@dataclass(frozen=True)
class EvaluatorConfiguration(ThresholdConfiguration):
    """
    Confidence and [IoU ↗](../../metrics/iou.md) threshold configuration for the pre-built
    2D Instance Segmentation workflow.
    Specify a confidence and IoU threshold to apply to all classes.
    """
