from typing import List
from typing import Optional

from pydantic import validate_arguments

from kolena._api.v1.workflow import WorkflowType
from kolena._utils.validators import ValidatorConfig
from kolena.detection import TestImage
from kolena.detection._datatypes import TestImageDataFrame
from kolena.detection._internal import BaseTestCase


class TestCase(BaseTestCase):
    """
    A test case holds a set of images to compute performance metrics against.

    :param name: name of the test case to create or load
    :param version: optionally specify the version of the test case to load. When absent, the latest version is loaded.
        Ignored when creating new test cases
    :param description: optionally specify a description for a newly created test case. For existing test cases, this
        description can be edited via :meth:`kolena.detection.TestCase.edit`
    :param images: optionally provide a list of images and associated ground truths to populate a new test case. For
        existing test cases, images can be edited via :meth:`kolena.detection.TestCase.edit`. Images must be registered
        ahead of time with :meth:`kolena.detection.register_dataset`
    """

    _TestImageClass = TestImage
    _TestImageDataFrameClass = TestImageDataFrame

    _workflow = WorkflowType.DETECTION

    @validate_arguments(config=ValidatorConfig)
    def __init__(
        self,
        name: str,
        version: Optional[int] = None,
        description: Optional[str] = None,
        images: Optional[List[_TestImageClass]] = None,
        reset: bool = False,
    ):
        super().__init__(name, WorkflowType.DETECTION, version, description, images, reset)
