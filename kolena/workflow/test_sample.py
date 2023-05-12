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
import copy
from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

from pydantic import StrictBool
from pydantic import StrictFloat
from pydantic import StrictInt
from pydantic import StrictStr
from pydantic.dataclasses import dataclass

from kolena._utils.validators import ValidatorConfig
from kolena.workflow._datatypes import DataType
from kolena.workflow._datatypes import TypedDataObject
from kolena.workflow._validators import safe_issubclass
from kolena.workflow._validators import validate_field
from kolena.workflow._validators import validate_metadata_dict
from kolena.workflow.asset import ImageAsset

#: Type of the ``metadata`` field that can be included on :class:`kolena.workflow.TestSample` definitions. String
#: (``str``) keys and scalar values (``int``, ``float``, ``str``, ``bool``, ``None``) as well as scalar list values are
#: permitted.
Metadata = Dict[
    str,
    Union[
        None,
        # prevent coercion of values in metadata -- see: https://pydantic-docs.helpmanual.io/usage/types/#strict-types
        StrictStr,
        StrictFloat,
        StrictInt,
        StrictBool,
        # Pydantic's StrictX doesn't play nicely with deserialization (e.g. isinstance("a string", StrictStr) => False)
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
_METADATA_KEY = "metadata"


class _TestSampleType(DataType):
    IMAGE = "IMAGE"
    TEXT = "TEXT"
    VIDEO = "VIDEO"
    DOCUMENT = "DOCUMENT"
    COMPOSITE = "COMPOSITE"
    CUSTOM = "CUSTOM"

    @staticmethod
    def _data_category() -> str:
        return "TEST_SAMPLE"


@dataclass(frozen=True, config=ValidatorConfig)
class TestSample(TypedDataObject[_TestSampleType], metaclass=ABCMeta):
    """
    The inputs to a model.

    Test samples can be customized as necessary for a workflow by extending this class or one of the built-in test
    sample types.

    Extensions to the ``TestSample`` class may define a ``metadata`` field of type
    :data:`kolena.workflow.test_sample.Metadata` containing a dictionary of scalar properties associated with the test
    sample, intended for use when sorting or filtering test samples.

    Kolena handles the ``metadata`` field differently from other test sample fields. Updates to the ``metadata`` object
    for a given test sample are merged with previously uploaded metadata. As such, ``metadata`` for a given test sample
    within a test case is **not** immutable, and should **not** be relied on when an implementation of
    :class:`kolena.workflow.Model` computes inferences, or when an implementation of :class:`kolena.workflow.Evaluator`
    evaluates metrics.
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
    A test sample composed of multiple basic :class:`TestSample` elements.

    An example application would be each test sample is a pair of face images, and the goal is to predict whether the
    two images are of the same person. For this use-case the test sample can be defined as:

    .. code-block:: python

        class FacePairSample(Composite):
            source: Image
            target: Image

    To facilitate visualization for this kind of use cases, see usage of :class:`kolena.workflow.GroundTruth` and
    :class:`kolena.workflow.Inference`.
    """

    @classmethod
    def _data_type(cls) -> _TestSampleType:
        return _TestSampleType.COMPOSITE


@dataclass(frozen=True, config=ValidatorConfig)
class Image(TestSample):
    """An image located in a cloud bucket or served at a URL."""

    locator: str

    @classmethod
    def _data_type(cls) -> _TestSampleType:
        return _TestSampleType.IMAGE


@dataclass(frozen=True, config=ValidatorConfig)
class ImagePair(Composite):
    """Two images."""

    a: Image
    b: Image


@dataclass(frozen=True, config=ValidatorConfig)
class Text(TestSample):
    """A text snippet."""

    text: str

    @classmethod
    def _data_type(cls) -> _TestSampleType:
        return _TestSampleType.TEXT


@dataclass(frozen=True, config=ValidatorConfig)
class ImageText(Composite):
    """An image paired with a text snippet."""

    image: Image
    text: Text


# NOTE: declare BaseVideo as separate class for extension -- default fields in main Video class prevent extension with
#  non-default fields
@dataclass(frozen=True, config=ValidatorConfig)
class BaseVideo(TestSample):
    """A video clip located in a cloud bucket or served at a URL."""

    #: URL (e.g. S3, HTTPS) of the video file
    locator: str

    @classmethod
    def _data_type(cls) -> _TestSampleType:
        return _TestSampleType.VIDEO


@dataclass(frozen=True, config=ValidatorConfig)
class Video(BaseVideo):
    """A video clip located in a cloud bucket or served at a URL."""

    #: Optionally provide asset locator for custom video thumbnail
    thumbnail: Optional[ImageAsset] = None

    #: Optionally specify start time of video snippet, in seconds
    start: Optional[float] = None

    #: Optionally specify end time of video snippet, in seconds
    end: Optional[float] = None

    def __post_init__(self) -> None:
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError(f"Specified start time '{self.start}' is after specified end time '{self.end}'")
        if self.start is not None and self.end is not None and (self.start < 0 or self.end < 0):
            raise ValueError(f"Specified start time '{self.start}' and end time '{self.end}' must be non-negative")


@dataclass(frozen=True, config=ValidatorConfig)
class Document(TestSample):
    """A remotely linked document, e.g. PDF or TXT file."""

    #: URL (e.g. S3, HTTPS) of the document
    locator: str

    @classmethod
    def _data_type(cls) -> _TestSampleType:
        return _TestSampleType.DOCUMENT


_TEST_SAMPLE_BASE_TYPES = [Composite, Image, Text, BaseVideo, Document]


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
    fields_by_name = copy.copy(getattr(test_sample_type, "__annotations__", {}))
    is_composite = issubclass(test_sample_type, Composite)
    for field_name, field_value in fields_by_name.items():
        if field_name == "metadata":
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
    for k, v in getattr(test_sample_type, "__annotations__", {}).items():
        if k != "metadata":
            if safe_issubclass(v, TestSample):
                composite_fields.append(k)

    return composite_fields
