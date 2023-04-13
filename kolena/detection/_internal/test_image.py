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
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar

from pydantic import validate_arguments

from kolena._utils.frozen import Frozen
from kolena._utils.validators import ValidatorConfig
from kolena.detection._internal.metadata import MetadataElement


TestImageType = TypeVar("TestImageType")


class BaseTestImage(ABC, Frozen):
    """Base class for all TestImage classes"""

    #: Bucket locator for the provided test sample, e.g. ``gs://my-bucket/path/to/image.png``.
    locator: str

    #: Dataset this test image belongs to. Empty when unspecified.
    dataset: str

    #: Metadata associated with this test image. Surfaced during test runs.
    metadata: Dict[str, MetadataElement]

    @validate_arguments(config=ValidatorConfig)
    def __init__(
        self,
        locator: str,
        dataset: Optional[str] = None,
        metadata: Optional[Dict[str, MetadataElement]] = None,
    ):
        self.locator = locator
        self.dataset = dataset or ""
        self.metadata = metadata or {}

    @classmethod
    @abstractmethod
    def _meta_keys(cls) -> List[str]:
        pass

    @classmethod
    @abstractmethod
    def _from_record(cls, record: Any) -> "BaseTestImage":
        pass

    @classmethod
    @abstractmethod
    def _to_record(cls, image: "BaseTestImage") -> Tuple:
        pass
