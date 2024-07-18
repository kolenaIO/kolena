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
Test samples are the inputs to your models when testing.

For example, for a model that processes specific regions within a larger image, its test sample may be defined:

```python
from dataclasses import dataclass

from kolena.workflow import Image
from kolena.workflow.annotation import BoundingBox

@dataclass(frozen=True)
class ImageWithRegion(Image):
    region: BoundingBox

example = ImageWithRegion(
    locator="s3://my-bucket/example-image.png",  # field from Image base class
    region=BoundingBox(top_left=(0, 0), bottom_right=(100, 100)),
)
```

!!! note "Versioning for `locator` files"

    Kolena supports versioning for files stored in Amazon S3 or Google Cloud Storage. Simply enable versioning on your
    [S3](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Versioning.html) or
    [GCS](https://cloud.google.com/storage/docs/object-versioning)
    bucket and pass the `versionId` or `generation` as a part of the `locator`:

    - S3 (using `versionId`): `s3://my-bucket/example-image.png?versionId=Bv38GKqEKxwr_HYTEXYEx6TQG_4.LkAX`
    - GCS (using `generation`): `gs://my-bucket/example-image.png?generation=1701352005168905`
"""
import copy
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

from kolena._utils.datatypes import DataCategory
from kolena._utils.datatypes import DataType
from kolena._utils.datatypes import TypedDataObject
from kolena._utils.pydantic_v1 import StrictBool
from kolena._utils.pydantic_v1 import StrictFloat
from kolena._utils.pydantic_v1 import StrictInt
from kolena._utils.pydantic_v1 import StrictStr
from kolena._utils.pydantic_v1.dataclasses import dataclass
from kolena._utils.validators import ValidatorConfig
from kolena.workflow._validators import get_data_object_field_types
from kolena.workflow._validators import safe_issubclass
from kolena.workflow._validators import validate_field
from kolena.workflow._validators import validate_metadata_dict
from kolena.workflow.asset import ImageAsset

Metadata = Dict[
    str,
    Union[
        None,
        # prevent coercion of values in metadata -- see:
        # https://pydantic-docs.helpmanual.io/usage/types/#strict-types
        StrictStr,
        StrictFloat,
        StrictInt,
        StrictBool,
        # Pydantic's StrictX doesn't play nicely with deserialization (e.g. isinstance("a string", StrictStr) =>
        # False)
        #  -- include base scalar types as fallbacks for this purpose
        str,
        float,
        int,
        bool,
        List[
            Union[
                None,
                StrictStr,
                StrictFloat,
                StrictInt,
                StrictBool,
                str,
                float,
                int,
                bool,
            ]
        ],
    ],
]
"""
Type of the `metadata` field that can be included on [`TestSample`][kolena.workflow.TestSample] definitions. String
(`str`) keys and scalar values (`int`, `float`, `str`, `bool`, `None`) as well as scalar list values are permitted.

```python
from dataclasses import dataclass, field
from kolena.workflow import Image, Metadata

@dataclass(frozen=True)
class ImageWithMetadata(Image):
    metadata: Metadata = field(default_factory=dict)
```
"""

_METADATA_KEY = "metadata"


class _TestSampleType(DataType):
    IMAGE = "IMAGE"
    TEXT = "TEXT"
    VIDEO = "VIDEO"
    DOCUMENT = "DOCUMENT"
    COMPOSITE = "COMPOSITE"
    POINT_CLOUD = "POINT_CLOUD"
    AUDIO = "AUDIO"
    CUSTOM = "CUSTOM"

    @staticmethod
    def _data_category() -> DataCategory:
        return DataCategory.TEST_SAMPLE


@dataclass(frozen=True, config=ValidatorConfig)
class TestSample(TypedDataObject[_TestSampleType], metaclass=ABCMeta):
    """
    The inputs to a model.

    Test samples can be customized as necessary for a workflow by extending this class or one of the built-in test
    sample types.

    Extensions to the `TestSample` class may define a `metadata` field of type
    [`Metadata`][kolena.workflow.test_sample.Metadata] containing a dictionary of scalar properties associated with the
    test sample, intended for use when sorting or filtering test samples.

    Kolena handles the `metadata` field differently from other test sample fields. Updates to the `metadata` object
    for a given test sample are merged with previously uploaded metadata. As such, `metadata` for a given test sample
    within a test case is **not** immutable, and should **not** be relied on when an implementation of
    [`Model`][kolena.workflow.Model] computes inferences, or when an implementation of
    [`Evaluator`][kolena.workflow.Evaluator] evaluates metrics.
    """

    @staticmethod
    def _data_type() -> _TestSampleType:
        return _TestSampleType.CUSTOM

    def _to_dict(self) -> Dict[str, Any]:
        base_dict = super()._to_dict()
        base_dict.pop(_METADATA_KEY, None)
        return base_dict

    def _to_metadata_dict(self) -> Metadata:
        base_dict = super()._to_dict()
        return base_dict.pop(_METADATA_KEY, {})


@dataclass(frozen=True, config=ValidatorConfig)
class Composite(TestSample):
    """
    A test sample composed of multiple basic [`TestSample`][kolena.workflow.TestSample] elements.

    An example application would be each test sample is a pair of face images, and the goal is to predict whether the
    two images are of the same person. For this use-case the test sample can be defined as:

    ```python
    class FacePairSample(Composite):
        source: Image
        target: Image
    ```

    To facilitate visualization for this kind of use cases, see usage of [`GroundTruth`][kolena.workflow.GroundTruth]
    and [`Inference`][kolena.workflow.Inference].
    """

    @classmethod
    def _data_type(cls) -> _TestSampleType:
        return _TestSampleType.COMPOSITE


@dataclass(frozen=True, config=ValidatorConfig)
class Image(TestSample):
    """An image located in a cloud bucket or served at a URL."""

    locator: str
    """The URL of this image, using e.g. `s3`, `gs`, or `https` scheme (`s3://my-bucket/path/to/image.png`)."""

    @classmethod
    def _data_type(cls) -> _TestSampleType:
        return _TestSampleType.IMAGE


@dataclass(frozen=True, config=ValidatorConfig)
class ImagePair(Composite):
    """Two [`Image`s][kolena.workflow.Image] paired together."""

    a: Image
    """The left [`Image`][kolena.workflow.Image] in the image pair."""

    b: Image
    """The right [`Image`][kolena.workflow.Image] in the image pair."""


@dataclass(frozen=True, config=ValidatorConfig)
class Text(TestSample):
    """An inline text snippet."""

    text: str
    """The text snippet."""

    @classmethod
    def _data_type(cls) -> _TestSampleType:
        return _TestSampleType.TEXT


@dataclass(frozen=True, config=ValidatorConfig)
class ImageText(Composite):
    """An image paired with a text snippet."""

    image: Image
    """The [`Image`][kolena.workflow.Image] in this image-text pair."""

    text: Text
    """The text snippet in this image-text pair."""


# NOTE: declare BaseVideo as separate class for extension -- default fields in main Video class prevent extension with
#  non-default fields
@dataclass(frozen=True, config=ValidatorConfig)
class BaseVideo(TestSample):
    """A video clip located in a cloud bucket or served at a URL."""

    locator: str
    """URL (e.g. S3, HTTPS) of the video file."""

    @classmethod
    def _data_type(cls) -> _TestSampleType:
        return _TestSampleType.VIDEO


@dataclass(frozen=True, config=ValidatorConfig)
class Video(BaseVideo):
    """A video clip located in a cloud bucket or served at a URL."""

    locator: str
    """URL (e.g. S3, HTTPS) of the video file."""

    thumbnail: Optional[ImageAsset] = None
    """Optionally provide asset locator for custom video thumbnail."""

    start: Optional[float] = None
    """Optionally specify start time of video snippet, in seconds."""

    end: Optional[float] = None
    """Optionally specify end time of video snippet, in seconds."""

    def __post_init__(self) -> None:
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError(f"Specified start time '{self.start}' is after specified end time '{self.end}'")
        if self.start is not None and self.end is not None and (self.start < 0 or self.end < 0):
            raise ValueError(f"Specified start time '{self.start}' and end time '{self.end}' must be non-negative")


@dataclass(frozen=True, config=ValidatorConfig)
class Document(TestSample):
    """A remotely linked document, e.g. PDF or TXT file."""

    locator: str
    """URL (e.g. S3, HTTPS) of the document."""

    @classmethod
    def _data_type(cls) -> _TestSampleType:
        return _TestSampleType.DOCUMENT


@dataclass(frozen=True, config=ValidatorConfig)
class PointCloud(TestSample):
    """A pointcloud file located in a cloud bucket or served at a URL."""

    locator: str
    """The URL of the pointcloud file, using e.g. `s3`, `gs`, or `https` scheme (`s3://my-bucket/path/to/image.pcd`)."""

    @classmethod
    def _data_type(cls) -> _TestSampleType:
        return _TestSampleType.POINT_CLOUD


@dataclass(frozen=True, config=ValidatorConfig)
class Audio(TestSample):
    """An audio file located in a cloud bucket or served at a URL."""

    locator: str
    """URL (e.g. S3, HTTPS) of the audio file."""

    @classmethod
    def _data_type(cls) -> _TestSampleType:
        return _TestSampleType.AUDIO


_TEST_SAMPLE_BASE_TYPES = [Composite, Image, Text, BaseVideo, Document, PointCloud, Audio]


def _validate_test_sample_type(test_sample_type: Type[TestSample], recurse: bool = True) -> None:
    if not issubclass(test_sample_type, TestSample):
        raise ValueError(f"Test sample must subclass {TestSample.__name__}")

    if (
        not issubclass(test_sample_type, tuple(_TEST_SAMPLE_BASE_TYPES))
        and test_sample_type._data_type() is not _TestSampleType.CUSTOM
    ):
        supported_bases = ", ".join(t.__name__ for t in _TEST_SAMPLE_BASE_TYPES)
        raise ValueError(
            f"Test sample not extending a supported base must declare data_type {_TestSampleType.CUSTOM.value}.\n"
            f"Supported bases: {supported_bases}",
        )

    # TODO: this is structurally identical to implementation in validate_ground_truth_type and
    #  validate_metrics_test_sample_type -- share?
    fields_by_name = copy.copy(get_data_object_field_types(test_sample_type))
    is_composite = issubclass(test_sample_type, Composite)
    for field_name, field_value in fields_by_name.items():
        if field_name == _METADATA_KEY:
            validate_metadata_dict(field_value)
        # check composite test sample, keeping recurse to restrict 1-level composition for now
        elif is_composite and safe_issubclass(field_value, TestSample):
            if not recurse:
                raise ValueError(f"Nested composite test sample is not supported: {field_name}.")
            _validate_test_sample_type(field_value, recurse=False)
        else:
            validate_field(field_name, field_value)


def _get_composite_fields(test_sample_type: Type[TestSample]) -> List[str]:
    composite_fields = []
    for k, v in get_data_object_field_types(test_sample_type).items():
        if k != _METADATA_KEY:
            if safe_issubclass(v, TestSample):
                composite_fields.append(k)

    return composite_fields
