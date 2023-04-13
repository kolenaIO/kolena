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
"""
Metadata associated with a :class:`kolena.classification.TestImage`.

.. code-block:: python

    test_image = TestImage("s3://bucket/path/to/image.png", metadata=dict(
        input_landmarks=Landmarks(*landmarks),
        input_bounding_box=BoundingBox(*bounding_box),
        image_grayscale=Asset("s3://bucket/path/to/image_grayscale.png"),
    ))
"""
from kolena.detection._internal.metadata import Annotation
from kolena.detection._internal.metadata import Asset
from kolena.detection._internal.metadata import BoundingBox
from kolena.detection._internal.metadata import Landmarks
from kolena.detection._internal.metadata import MetadataElement

__all__ = [
    "Annotation",
    "BoundingBox",
    "Landmarks",
    "Asset",
    "MetadataElement",
]
