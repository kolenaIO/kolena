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
import dataclasses
import json
from abc import ABC
from collections import OrderedDict
from contextlib import contextmanager
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Type

import pandas as pd
from pydantic import validate_arguments

from kolena._api.v1.core import TestCase as CoreAPI
from kolena._api.v1.detection import TestCase as API
from kolena._api.v1.workflow import WorkflowType
from kolena._utils import krequests
from kolena._utils import log
from kolena._utils.batched_load import _BatchedLoader
from kolena._utils.batched_load import DFType
from kolena._utils.batched_load import init_upload
from kolena._utils.batched_load import upload_data_frame
from kolena._utils.consts import BatchSize
from kolena._utils.dataframes.validators import validate_df_schema
from kolena._utils.frozen import Frozen
from kolena._utils.instrumentation import WithTelemetry
from kolena._utils.serde import from_dict
from kolena._utils.validators import ValidatorConfig
from kolena.detection._internal import BaseTestImage
from kolena.errors import NotFoundError
from kolena.workflow._validators import assert_workflows_match


class BaseTestCase(ABC, Frozen, WithTelemetry):
    """
    A test case holds a set of images to compute performance metrics against.

    :param name: name of the test case to create or load
    :param version: optionally specify the version of the test case to load. When absent, the latest version is loaded.
        Ignored when creating new test cases
    :param description: optionally specify a description for a newly created test case. For existing test cases, this
        description can be edited via :meth:`kolena.detection.TestCase.edit`
    :param images: optionally provide a list of images and associated ground truths to populate a new test case. For
        existing test cases, images can be edited via :meth:`TestCase.edit`.
    """

    _TestImageClass: Type[BaseTestImage] = BaseTestImage
    _TestImageDataFrameClass: Type[DFType] = DFType

    #: The unique name of this test case. Cannot be changed after creation.
    name: str

    #: The version of this test case. A test case's version is automatically incremented whenever it is edited via
    #: :meth:`TestCase.edit`.
    version: int

    #: Free-form, human-readable description of this test case. Can be edited at any time via :meth:`TestCase.edit`.
    description: str

    _id: int
    _workflow: WorkflowType

    def __init__(
        self,
        name: str,
        workflow: WorkflowType,
        version: Optional[int] = None,
        description: Optional[str] = None,
        images: Optional[List[_TestImageClass]] = None,
        reset: bool = False,
    ):
        try:
            self._populate_from_other(self.load(name, version))
            if description is not None and self.description != description and not reset:
                log.warn("test case already exists, not updating description when reset=False")
            if images is not None:
                if self.version > 0 and not reset:
                    log.warn("not updating images for test case that has already been edited when reset=False")
                else:
                    self._hydrate(images, description)
        except NotFoundError:
            if version is not None:
                log.warn(f"creating new test case '{name}', ignoring provided version")
            self._populate_from_other(self._create(workflow, name, description, images))
        self._freeze()

    def _populate_from_other(self, other: "BaseTestCase") -> None:
        with self._unfrozen():
            self._id = other._id
            self.name = other.name
            self.version = other.version
            self.description = other.description
            self._workflow = other._workflow

    @classmethod
    def _from(cls, response: CoreAPI.EntityData) -> "BaseTestCase":
        obj = cls.__new__(cls)
        obj.name = response.name
        obj.version = response.version
        obj.description = response.description
        obj._id = response.id
        obj._workflow = WorkflowType(response.workflow)
        obj._freeze()
        return obj

    @classmethod
    def _create(
        cls,
        workflow: WorkflowType,
        name: str,
        description: Optional[str] = None,
        images: Optional[List[_TestImageClass]] = None,
    ) -> "BaseTestCase":
        """Create a new test case with the provided name."""
        request = CoreAPI.CreateRequest(name=name, description=description or "", workflow=workflow.value)
        res = krequests.post(endpoint_path=API.Path.CREATE.value, data=json.dumps(dataclasses.asdict(request)))
        krequests.raise_for_status(res)
        data = from_dict(data_class=CoreAPI.EntityData, data=res.json())
        obj = cls._create_from_data(data)
        log.info(f"created test case '{name}' (v{obj.version})")
        if images is not None:
            obj._hydrate(images)
        return obj

    @classmethod
    @validate_arguments(config=ValidatorConfig)
    def _load_by_name(cls, name: str, version: Optional[int] = None) -> CoreAPI.EntityData:
        """Load an existing test case with the provided name."""
        request = CoreAPI.LoadByNameRequest(name=name, version=version)
        res = krequests.put(endpoint_path=API.Path.LOAD_BY_NAME.value, data=json.dumps(dataclasses.asdict(request)))
        krequests.raise_for_status(res)
        obj = from_dict(data_class=CoreAPI.EntityData, data=res.json())
        log.info(f"loaded test case '{name}' (v{obj.version})")
        return obj

    @validate_arguments(config=ValidatorConfig)
    def _hydrate(self, images: List[_TestImageClass], description: Optional[str] = None) -> None:
        if len(images) == 0:
            log.warn("no images provided, unable to populate test case")
            return
        with self.edit(reset=True) as editor:
            if description is not None:
                editor.description(description)
            for image in images:
                editor.add(image)

    @classmethod
    def _create_from_data(cls, data: CoreAPI.EntityData) -> "BaseTestCase":
        assert_workflows_match(cls._workflow, data.workflow)
        return cls._from(data)

    @validate_arguments(config=ValidatorConfig)
    def load_images(self) -> List[_TestImageClass]:
        """Load all test images with their associated ground truths in this test case."""
        return list(self.iter_images())

    @validate_arguments(config=ValidatorConfig)
    def iter_images(self) -> Iterator[_TestImageClass]:
        """Iterate through all images with their associated ground truths in this test case."""
        log.info(f"loading test images for test case '{self.name}' (v{self.version})")
        init_request = CoreAPI.InitLoadContentsRequest(batch_size=BatchSize.LOAD_SAMPLES.value, test_case_id=self._id)
        for df in _BatchedLoader.iter_data(
            init_request=init_request,
            endpoint_path=API.Path.INIT_LOAD_IMAGES.value,
            df_class=self._TestImageDataFrameClass,
        ):
            for record in df.itertuples():
                yield self._TestImageClass._from_record(record)
        log.info(f"loaded test images for test case '{self.name}' (v{self.version})")

    @classmethod
    def create(
        cls,
        name: str,
        description: Optional[str] = None,
        images: Optional[List[_TestImageClass]] = None,
    ) -> "BaseTestCase":
        """
        Create a new test case with the provided name.

        :param name: the name of the new test case to create.
        :param description: optional free-form description of the test case to create.
        :param images: optionally specify a set of images to populate the test case.
        :return: the newly created test case.
        """
        return cls._create(cls._workflow, name, description, images)

    @classmethod
    def load(cls, name: str, version: Optional[int] = None) -> "BaseTestCase":
        """
        Load an existing test case with the provided name.
        :param name: the name of the test case to load.
        :param version: optionally specify a particular version of the test case to load. Defaults to the latest version
            when unset.
        :return: the loaded test case.
        """
        data = cls._load_by_name(name, version)
        return cls._create_from_data(data)

    class Editor:
        """
        Interface to edit a test case. Create with :meth:`TestCase.edit`.
        """

        _TestImageClass: Type[BaseTestImage] = BaseTestImage
        _images: Dict[str, BaseTestImage]
        _reset: bool
        _description: str
        _initial_description: str

        def __init__(self, description: str, reset: bool) -> None:
            self._edited = False
            self._reset = reset
            self._description = description
            self._initial_description = description
            self._images: Dict[str, BaseTestImage] = OrderedDict()

        @validate_arguments(config=ValidatorConfig)
        def description(self, description: str) -> None:
            """
            Update the description of this test case.

            :param description: the new test case description
            """
            if self._description == description:
                return
            self._description = description
            self._edited = True

        @validate_arguments(config=ValidatorConfig)
        def add(self, image: _TestImageClass) -> None:
            """
            Add a test image to the test case, targeting the ``ground_truths`` held by the image.
            When the test image already exists in the test case, its ground truth
            is overwritten.

            To filter the ground truths associated with a test image, see :meth:`kolena.detection.TestImage.filter`.

            :param image: the test image to add to the test case, holding corresponding ground truths
            """
            if image == self._images.get(image.locator, None):
                log.info(f"no op: {image.locator} already in test case")
                return
            self._images[image.locator] = image
            self._edited = True

        @validate_arguments(config=ValidatorConfig)
        def remove(self, image: _TestImageClass) -> None:
            """
            Remove the image from the test case.

            :param image: the test image to remove
            :raises KeyError: if the image is not in the test case
            """
            if image.locator not in self._images.keys():
                raise KeyError(f"unrecognized image: '{image.locator}' not in test case")
            self._images.pop(image.locator)
            self._edited = True

    @classmethod
    def _to_data_frame(cls, images: List[_TestImageClass]) -> _TestImageDataFrameClass:
        records = [cls._TestImageClass._to_record(image) for image in images]
        columns = cls._TestImageClass._meta_keys()
        df = pd.DataFrame(records, columns=columns)
        return cls._TestImageDataFrameClass(
            validate_df_schema(df, cls._TestImageDataFrameClass.get_schema(), trusted=True),
        )

    @contextmanager
    def edit(self, reset: bool = False) -> Iterator[Editor]:
        """
        Edit this test case in a context:

        .. code-block:: python

            with test_case.edit() as editor:
                # perform as many editing actions as desired
                editor.add(...)
                editor.remove(...)

        Changes are committed to the Kolena platform when the context is exited.

        :param reset: clear any and all test samples currently in the test case.
        """
        editor = self.Editor(self.description, reset)
        editor._TestImageClass = self._TestImageClass
        if not reset:
            for image in self.iter_images():
                editor.add(image)
            editor._edited = False

        yield editor

        # no-op contexts have no effect, do not bump version
        if not editor._edited:
            return

        log.info(f"editing test case '{self.name}' (v{self.version})")
        init_response = init_upload()
        df = self._to_data_frame(list(editor._images.values()))
        df_serialized = df.as_serializable()
        upload_data_frame(df=df_serialized, batch_size=BatchSize.UPLOAD_RECORDS.value, load_uuid=init_response.uuid)

        request = CoreAPI.CompleteEditRequest(
            test_case_id=self._id,
            current_version=self.version,
            description=editor._description,
            reset=editor._reset,
            uuid=init_response.uuid,
        )
        complete_res = krequests.put(
            endpoint_path=API.Path.COMPLETE_EDIT.value,
            data=json.dumps(dataclasses.asdict(request)),
        )
        krequests.raise_for_status(complete_res)
        test_case_data = from_dict(data_class=CoreAPI.EntityData, data=complete_res.json())
        self._populate_from_other(self._create_from_data(test_case_data))
        log.success(f"edited test case '{self.name}' (v{self.version})")
