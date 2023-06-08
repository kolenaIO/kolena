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
import json
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

from deprecation import deprecated
from pydantic import validate_arguments

from kolena._api.v1.detection import TestImage as API
from kolena._utils.batched_load import _BatchedLoader
from kolena._utils.consts import BatchSize
from kolena._utils.validators import ValidatorConfig
from kolena.detection._datatypes import TestImageDataFrame
from kolena.detection._internal import BaseTestImage
from kolena.detection._internal.metadata import _from_dict
from kolena.detection._internal.metadata import _to_dict
from kolena.detection._internal.metadata import MetadataElement
from kolena.detection.ground_truth import GroundTruth


class TestImage(BaseTestImage):
    """
    Test image comprising a single image locator.
    """

    locator: str
    """Bucket locator for the provided test sample, e.g. `gs://my-bucket/path/to/image.png`."""

    dataset: str
    """Dataset this test image belongs to. Empty when unspecified."""

    metadata: Dict[str, MetadataElement]
    """Metadata associated with this test image. Surfaced during test runs."""

    ground_truths: List[GroundTruth]
    """List of [`GroundTruth`][kolena.detection.ground_truth.GroundTruth] annotations associated with this image."""

    @validate_arguments(config=ValidatorConfig)
    def __init__(
        self,
        locator: str,
        dataset: Optional[str] = None,
        ground_truths: Optional[List[GroundTruth]] = None,
        metadata: Dict[str, MetadataElement] = None,
    ):
        super().__init__(locator, dataset, metadata)
        self.ground_truths = ground_truths or []

    def filter(self, predicate: Callable[[GroundTruth], bool]) -> "TestImage":
        """
        Return a copy of this test image with ground truths filtered to only those that match the provided predicate.

        :param predicate: Function accepting a [`GroundTruth`][kolena.detection.ground_truth.GroundTruth] and returning
            a boolean indicating whether or not to include the ground truth.
        :return: A new test image with ground truths filtered by the predicate.
        """
        return TestImage(**{**self._fields(), "ground_truths": [gt for gt in self.ground_truths if predicate(gt)]})

    @classmethod
    def _meta_keys(cls) -> List[str]:
        return ["locator", "dataset", "ground_truths", "metadata"]

    @classmethod
    def _from_record(cls, record: Any) -> "TestImage":
        return TestImage(
            record.locator,
            dataset=record.dataset,
            ground_truths=[GroundTruth._from_dict(gt_dict) for gt_dict in record.ground_truths or []],
            metadata=_from_dict(record.metadata),
        )

    @classmethod
    def _to_record(cls, image: "TestImage") -> Tuple[str, Optional[str], List[Dict[str, Any]], Dict[str, Any]]:
        return (image.locator, image.dataset, [gt._to_dict() for gt in image.ground_truths], _to_dict(image.metadata))

    # TODO: remove implementation in favor of Frozen.__eq__ once ground_truth ordering is ensured upstream
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) and {
            **self.__dict__,
            "ground_truths": self._sort_ground_truths(self.ground_truths),
        } == {**other.__dict__, "ground_truths": self._sort_ground_truths(other.ground_truths)}

    @staticmethod
    def _sort_ground_truths(ground_truths: List[GroundTruth]) -> List[GroundTruth]:
        return sorted(ground_truths, key=lambda gt: json.dumps(gt._to_dict(), sort_keys=True))


@deprecated(details="use `TestCase.load_images`", deprecated_in="0.26.0")
@validate_arguments(config=ValidatorConfig)
def load_images(dataset: Optional[str] = None) -> List[TestImage]:
    """
    !!! warning "Deprecated: since `0.26.0`"
        Please use [`TestCase.load_images`][kolena.detection._internal.test_case.BaseTestCase.load_images] instead.

    Load a list of [`TestImage`][kolena.detection.TestImage] samples registered in Kolena.

    :param dataset: Optionally specify the single dataset to be retrieved. By default, images from all
        datasets are returned.
    """
    return list(iter_images(dataset))


@deprecated(details="use `TestCase.iter_images`", deprecated_in="0.26.0")
@validate_arguments(config=ValidatorConfig)
def iter_images(dataset: Optional[str] = None) -> Iterator[TestImage]:
    """
    !!! warning "Deprecated: since `0.26.0`"
        Please use [`TestCase.iter_images`][kolena.detection._internal.test_case.BaseTestCase.iter_images] instead.

    Return iterator over [`TestImage`][kolena.detection.TestImage] samples registered in Kolena. Images are lazily
    loaded in chunks to facilitate working with large datasets that are cumbersome to hold in memory.

    :param dataset: Optionally specify the single dataset to be retrieved. By default, images from all
        datasets are returned.
    """
    init_request = API.InitLoadImagesRequest(dataset=dataset, batch_size=BatchSize.LOAD_RECORDS.value)
    for df in _BatchedLoader.iter_data(
        init_request=init_request,
        endpoint_path=API.Path.INIT_LOAD_IMAGES.value,
        df_class=TestImageDataFrame,
    ):
        for record in df.itertuples():
            yield TestImage._from_record(record)
