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
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from pydantic import validate_arguments

from kolena import detection
from kolena._utils.validators import ValidatorConfig
from kolena.detection._internal.metadata import _from_dict
from kolena.detection._internal.metadata import MetadataElement
from kolena.detection._internal.test_image import BaseTestImage
from kolena.detection.ground_truth import ClassificationLabel
from kolena.detection.ground_truth import GroundTruth


class TestImage(BaseTestImage):
    """
    An image with associated ground truth labels for testing.
    """

    #: Pointer to the bucket location of this image, e.g. ``gs://my-bucket/my-dataset/example.png``.
    locator: str

    #: The source dataset this image belongs to.
    dataset: str

    #: Zero or more ground truth labels for this image. For binary classifiers, an arbitrary string such as ``positive``
    #: may be used. Not surfaced during testing.
    labels: List[str]

    #: Arbitrary metadata associated with this image. This metadata is surfaced during testing and may be used as model
    #: inputs as necessary.
    #:
    #: Certain metadata values can be visualized in the web platform when viewing results:
    #:
    #: * :class:`kolena.classification.metadata.Annotation` objects are overlaid on the main image
    #: * :class:`kolena.classification.metadata.Asset` objects containing locators pointing to images, e.g.
    #:   ``gs://my-bucket/my-dataset/example-1channel.png``, are displayed
    #:
    #: See :mod:`kolena.classification.metadata` documentation for more details.
    metadata: Dict[str, MetadataElement]

    @validate_arguments(config=ValidatorConfig)
    def __init__(
        self,
        locator: str,
        dataset: Optional[str] = None,
        labels: Optional[List[str]] = None,
        metadata: Optional[Dict[str, MetadataElement]] = None,
    ):
        super().__init__(locator, dataset, metadata)
        self.labels = labels or []

    def filter(self, predicate: Callable[[str], bool]) -> "TestImage":
        """
        Return a copy of this test image with ground truth labels filtered to only those that match the provided
        predicate.

        :param predicate: function accepting a string label and returning a boolean indicating whether or not to include
            the ground truth label
        :return: a new test image with labels filtered by the predicate
        """
        return TestImage(**{**self._fields(), "labels": [label for label in self.labels if predicate(label)]})

    @classmethod
    def _meta_keys(cls) -> List[str]:
        return detection.TestImage._meta_keys()

    @classmethod
    def _from_record(cls, record: Any) -> "TestImage":
        return TestImage(
            record.locator,
            dataset=record.dataset,
            labels=[GroundTruth._from_dict(gt_dict).label for gt_dict in record.ground_truths or []],
            metadata=_from_dict(record.metadata),
        )

    @classmethod
    def _to_record(cls, image: "TestImage") -> Tuple[str, Optional[str], List[Dict[str, Any]], Dict[str, Any]]:
        return detection.TestImage._to_record(image._to_detection())

    def _to_detection(self) -> detection.TestImage:
        return detection.TestImage(
            locator=self.locator,
            dataset=self.dataset,
            ground_truths=[ClassificationLabel(label) for label in self.labels],
            metadata=self.metadata,
        )

    # TODO: remove implementation in favor of Frozen.__eq__ once label ordering is ensured upstream
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) and {**self.__dict__, "labels": sorted(self.labels)} == {
            **other.__dict__,
            "labels": sorted(other.labels),
        }
