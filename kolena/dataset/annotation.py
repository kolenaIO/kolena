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
from kolena.workflow.annotation import BitmapMask
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import BoundingBox3D
from kolena.workflow.annotation import ClassificationLabel
from kolena.workflow.annotation import Keypoints
from kolena.workflow.annotation import Label
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import LabeledBoundingBox3D
from kolena.workflow.annotation import LabeledPolygon
from kolena.workflow.annotation import LabeledTimeSegment
from kolena.workflow.annotation import Polygon
from kolena.workflow.annotation import Polyline
from kolena.workflow.annotation import ScoredBoundingBox
from kolena.workflow.annotation import ScoredBoundingBox3D
from kolena.workflow.annotation import ScoredClassificationLabel
from kolena.workflow.annotation import ScoredLabel
from kolena.workflow.annotation import ScoredLabeledBoundingBox
from kolena.workflow.annotation import ScoredLabeledBoundingBox3D
from kolena.workflow.annotation import ScoredLabeledPolygon
from kolena.workflow.annotation import ScoredLabeledTimeSegment
from kolena.workflow.annotation import ScoredPolygon
from kolena.workflow.annotation import ScoredTimeSegment
from kolena.workflow.annotation import SegmentationMask
from kolena.workflow.annotation import TimeSegment

__all__ = [
    "BoundingBox",
    "LabeledBoundingBox",
    "ScoredBoundingBox",
    "ScoredLabeledBoundingBox",
    "Polygon",
    "LabeledPolygon",
    "ScoredPolygon",
    "ScoredLabeledPolygon",
    "Keypoints",
    "Polyline",
    "BoundingBox3D",
    "LabeledBoundingBox3D",
    "ScoredBoundingBox3D",
    "ScoredLabeledBoundingBox3D",
    "SegmentationMask",
    "BitmapMask",
    "Label",
    "ScoredLabel",
    "ClassificationLabel",
    "ScoredClassificationLabel",
    "TimeSegment",
    "LabeledTimeSegment",
    "ScoredTimeSegment",
    "ScoredLabeledTimeSegment",
]
