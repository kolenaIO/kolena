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
from typing import List
from typing import Optional

from pydantic import validate_arguments

from kolena._api.v1.workflow import WorkflowType
from kolena._utils.validators import ValidatorConfig
from kolena.classification import TestImage
from kolena.detection._datatypes import TestImageDataFrame
from kolena.detection._internal import BaseTestCase


class TestCase(BaseTestCase):
    """
    A test case is the base grouping of test data in the Kolena platform.

    Fundamentally, a test case can be thought of as a benchmark dataset. Metrics are computed for each test case.

    A test case may be as large or as small as necessary. A test case may have millions of images for high-level results
    across a large population. Alternatively, a test case may have only one or a handful of images for laser-focus on a
    specific scenario.

    :param name: the name of the test case. If a test case by this name already exists, that test case is loaded
    :param version: optionally specify the version of the test case to load. Ignored when a test case by the
        provided name does not already exist
    :param description: optionally specify a description for the new test case. Ignored when a test case with the
        provided name already exists
    :param images: optionally provide a list of :class:`kolena.classification.TestImage` images used to seed a new test
        case. Ignored when a test case with the provided name already exists
    """

    _TestImageClass = TestImage
    _TestImageDataFrameClass = TestImageDataFrame

    _workflow = WorkflowType.CLASSIFICATION

    @validate_arguments(config=ValidatorConfig)
    def __init__(
        self,
        name: str,
        version: Optional[int] = None,
        description: Optional[str] = None,
        images: Optional[List[_TestImageClass]] = None,
        reset: bool = False,
    ):
        super().__init__(name, WorkflowType.CLASSIFICATION, version, description, images, reset)
