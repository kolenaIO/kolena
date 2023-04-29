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
from dataclasses import dataclass
from typing import Optional

from kolena.workflow import Composite
from kolena.workflow import DataObject
from kolena.workflow import Image
from kolena.workflow import Metadata
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.asset import ImageAsset


class ImageTriplet(Composite):
    a: Image
    b: Image
    c: Image
    d: str
    e: Optional[ImageAsset]
    metadata: Metadata


@dataclass(frozen=True)
class ComplexBoundingBox(DataObject):
    a: int
    b: float
    c: BoundingBox


@dataclass(frozen=True)
class NestedComplexBoundingBox(DataObject):
    a: ComplexBoundingBox
    b: int
