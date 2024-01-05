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
Assets are additional files linked to the [`TestSample`][kolena.workflow.TestSample],
[`GroundTruth`][kolena.workflow.GroundTruth], or [`Inference`][kolena.workflow.Inference] objects for your workflow.
Assets can be visualized in the Kolena Studio when exploring your test cases or model results.

The following asset types are available:

- [`ImageAsset`][kolena.workflow.asset.ImageAsset]
- [`PlainTextAsset`][kolena.workflow.asset.PlainTextAsset]
- [`BinaryAsset`][kolena.workflow.asset.BinaryAsset]
- [`PointCloudAsset`][kolena.workflow.asset.PointCloudAsset]
- [`VideoAsset`][kolena.workflow.asset.VideoAsset]

"""
from kolena.asset import Asset
from kolena.asset import BaseVideoAsset
from kolena.asset import BinaryAsset
from kolena.asset import ImageAsset
from kolena.asset import PlainTextAsset
from kolena.asset import PointCloudAsset
from kolena.asset import VideoAsset

__all__ = ["Asset", "ImageAsset", "PlainTextAsset", "BinaryAsset", "PointCloudAsset", "BaseVideoAsset", "VideoAsset"]
