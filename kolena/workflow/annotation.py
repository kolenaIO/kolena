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
"""
Annotations are visualized in Kolena as overlays on top of [`TestSample`][kolena.workflow.TestSample] objects.

| Annotation | Valid [`TestSample`][kolena.workflow.TestSample] Types |
| --- | --- |
| [`BoundingBox`][kolena.workflow.annotation.BoundingBox] | [`Image`][kolena.workflow.Image], [`Video`][kolena.workflow.Video] |
| [`BoundingBox3D`][kolena.workflow.annotation.BoundingBox3D] | [`PointCloud`][kolena.workflow.PointCloud] |
| [`Polygon`][kolena.workflow.annotation.Polygon] | [`Image`][kolena.workflow.Image], [`Video`][kolena.workflow.Video] |
| [`Polyline`][kolena.workflow.annotation.Polyline] | [`Image`][kolena.workflow.Image], [`Video`][kolena.workflow.Video] |
| [`Keypoints`][kolena.workflow.annotation.Keypoints] | [`Image`][kolena.workflow.Image], [`Video`][kolena.workflow.Video] |
| [`SegmentationMask`][kolena.workflow.annotation.SegmentationMask] | [`Image`][kolena.workflow.Image], [`Video`][kolena.workflow.Video] |
| [`BitmapMask`][kolena.workflow.annotation.BitmapMask] | [`Image`][kolena.workflow.Image], [`Video`][kolena.workflow.Video] |
| [`Label`][kolena.workflow.annotation.Label] | [`Text`][kolena.workflow.Text], [`Document`][kolena.workflow.Document], [`Image`][kolena.workflow.Image], [`PointCloud`][kolena.workflow.PointCloud], [`Audio`][kolena.workflow.Audio], [`Video`][kolena.workflow.Video] |
| [`TimeSegment`][kolena.workflow.annotation.TimeSegment] | [`Audio`][kolena.workflow.Audio], [`Video`][kolena.workflow.Video] |

For example, when viewing images in the Studio, any annotations (such as lists of
[`BoundingBox`][kolena.workflow.annotation.BoundingBox] objects) present in the
[`TestSample`][kolena.workflow.TestSample], [`GroundTruth`][kolena.workflow.GroundTruth],
[`Inference`][kolena.workflow.Inference], or [`MetricsTestSample`][kolena.workflow.MetricsTestSample] objects are
rendered on top of the image.
"""  # noqa: E501
from kolena.annotation import Annotation
from kolena.annotation import BitmapMask
from kolena.annotation import BoundingBox
from kolena.annotation import BoundingBox3D
from kolena.annotation import ClassificationLabel
from kolena.annotation import Keypoints
from kolena.annotation import Label
from kolena.annotation import LabeledBoundingBox
from kolena.annotation import LabeledBoundingBox3D
from kolena.annotation import LabeledPolygon
from kolena.annotation import LabeledTimeSegment
from kolena.annotation import Polygon
from kolena.annotation import Polyline
from kolena.annotation import ScoredBoundingBox
from kolena.annotation import ScoredBoundingBox3D
from kolena.annotation import ScoredClassificationLabel
from kolena.annotation import ScoredLabel
from kolena.annotation import ScoredLabeledBoundingBox
from kolena.annotation import ScoredLabeledBoundingBox3D
from kolena.annotation import ScoredLabeledPolygon
from kolena.annotation import ScoredLabeledTimeSegment
from kolena.annotation import ScoredPolygon
from kolena.annotation import ScoredTimeSegment
from kolena.annotation import SegmentationMask
from kolena.annotation import TimeSegment

__all__ = [
    "Annotation",
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
