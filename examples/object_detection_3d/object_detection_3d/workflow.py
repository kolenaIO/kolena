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
from typing import List
from typing import Optional

from pydantic.dataclasses import dataclass

from kolena.workflow import define_workflow
from kolena.workflow import GroundTruth as BaseGroundTruth
from kolena.workflow import Image
from kolena.workflow import Inference as BaseInference
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import LabeledBoundingBox3D
from kolena.workflow.annotation import ScoredLabeledBoundingBox
from kolena.workflow.annotation import ScoredLabeledBoundingBox3D
from kolena.workflow.asset import ImageAsset
from kolena.workflow.asset import PointCloudAsset


@dataclass(frozen=True)
class TestSample(Image):
    velodyne: PointCloudAsset
    velodyne_to_camera_transformation: List[float]  # row-major (4X4)
    camera_rectification: List[float]  # row-major (4X4)
    image_projection: List[float]  # row-major (4x4)
    right: Optional[ImageAsset] = None

    def __post_init__(self) -> None:
        if len(self.velodyne_to_camera_transformation) != 16:
            raise ValueError(
                f"{type(self).__name__} must have valid 16 floats \
                               in velodyne_to_camera_transformation \
                               ({len(self.velodyne_to_camera_transformation)} provided)",
            )
        if len(self.image_projection) != 16:
            raise ValueError(
                f"{type(self).__name__} must have valid 16 floats \
                               in image_projection ({len(self.image_projection)} provided)",
            )
        if len(self.image_projection) != 16:
            raise ValueError(
                f"{type(self).__name__} must have valid 16 floats \
                               in image_projection ({len(self.image_projection)} provided)",
            )


@dataclass(frozen=True)
class AnnotatedBoundingBox(LabeledBoundingBox):
    difficulty: str


@dataclass(frozen=True)
class AnnotatedBoundingBox3D(LabeledBoundingBox3D):
    """
    #Values    Name      Description
    ----------------------------------------------------------------------------
    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                        'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                        'Misc' or 'DontCare'
    1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                        truncated refers to the object leaving image boundaries
    1    occluded     Integer (0,1,2,3) indicating occlusion state:
                        0 = fully visible, 1 = partly occluded
                        2 = largely occluded, 3 = unknown
    1    alpha        Observation angle of object, ranging [-pi..pi]
    4    bbox         2D bounding box of object in the image (0-based index):
                        contains left, top, right, bottom pixel coordinates
    3    dimensions   3D object dimensions: height, width, length (in meters)
    3    location     3D object location x,y,z in camera coordinates (in meters)
    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
    1    score        Only for results: Float, indicating confidence in
                        detection, needed for p/r curves, higher is better.
    """

    truncated: float
    occluded: int
    alpha: float
    difficulty: str


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    total_objects: int
    n_pedestrian: int
    n_car: int
    n_cyclist: int
    n_easy: int
    n_moderate: int
    n_hard: int
    n_unknown: int
    bboxes_2d: List[LabeledBoundingBox]
    bboxes_3d: List[AnnotatedBoundingBox3D]


@dataclass(frozen=True)
class Inference(BaseInference):
    bboxes_2d: List[ScoredLabeledBoundingBox]
    bboxes_3d: List[ScoredLabeledBoundingBox3D]


_workflow, TestCase, TestSuite, Model = define_workflow(
    "KITTI 3D Object Detection",
    TestSample,
    GroundTruth,
    Inference,
)
