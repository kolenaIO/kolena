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
from contextlib import contextmanager
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Set
from typing import Union

import numpy as np
import pandas as pd
from pydantic import validate_arguments
from pydantic.dataclasses import dataclass

from kolena._api.v1.batched_load import BatchedLoad as LoadAPI
from kolena._api.v1.fr import TestImages as API
from kolena._utils import krequests
from kolena._utils import log
from kolena._utils.batched_load import _BatchedLoader
from kolena._utils.batched_load import init_upload
from kolena._utils.batched_load import upload_data_frame
from kolena._utils.consts import BatchSize
from kolena._utils.dataframes.validators import validate_df_schema
from kolena._utils.uninstantiable import Uninstantiable
from kolena._utils.validators import ValidatorConfig
from kolena.fr import TestCase
from kolena.fr import TestSuite
from kolena.fr.datatypes import TEST_IMAGE_COLUMNS
from kolena.fr.datatypes import TestImageDataFrame
from kolena.fr.datatypes import TestImageDataFrameSchema
from kolena.fr.datatypes import TestImageRecord


class TestImages(Uninstantiable[None]):
    @classmethod
    @validate_arguments(config=ValidatorConfig)
    def load(
        cls,
        data_source: Optional[Union[str, TestSuite, TestSuite.Data, TestCase, TestCase.Data]] = None,
        include_augmented: bool = False,
    ) -> TestImageDataFrame:
        """
        Load a DataFrame describing images registered in the Kolena platform.

        :param data_source: optionally specify the single data source to be retrieved, e.g. ``"my-data-source"``.
            Alternatively, provide a :class:`kolena.fr.TestSuite` or :class:`kolena.fr.TestCase` as source.
            If no argument is provided, all images registered using :meth:`TestImages.register` are returned
        :param include_augmented: optionally specify that augmented images should be returned. By default, only
            original images are returned. Ignored when test case or test suite is provided as ``data_source``
        """
        log.info("loading test images")
        return _BatchedLoader.concat(
            cls.iter(data_source=data_source, include_augmented=include_augmented),
            TestImageDataFrame,
        )

    @dataclass(config=ValidatorConfig)
    class _Registrar:
        records: List[TestImageRecord]
        locators: Set[str]

    class Registrar(Uninstantiable[_Registrar]):
        @validate_arguments(config=ValidatorConfig)
        def add(
            self,
            locator: str,
            data_source: str,
            width: int,
            height: int,
            bounding_box: Optional[np.ndarray] = None,
            landmarks: Optional[np.ndarray] = None,
            tags: Optional[Dict[str, str]] = None,
        ) -> None:
            """
            Add a new image to Kolena. If the provided locator is already registered with the platform, its metadata
            will be updated.

            :param locator: bucket locator for the provided image, e.g. ``s3://bucket-name/path/to/image.jpg``
            :param data_source: name of the source for the image being registered
            :param width: width in pixels of the image being registered
            :param height: height in pixels of the image being registered
            :param bounding_box: optional 4-element array specifying the ground truth bounding box for this image, of
                the form ``[top_left_x, top_left_y, bottom_right_x, bottom_right_y]``
            :param landmarks: optional 10-element array specifying (x, y) coordinates for five facial landmarks of the
                form ``[left_eye_{x,y}, right_eye_{x,y}, nose_{x,y}, left_mouth_{x,y}, right_mouth_{x,y}]``
            :param tags: tags to associate with the image, of the form ``{category: value}``
            """
            if locator in self.data.locators:
                raise ValueError(f"duplicate locator: {locator}")
            self.data.locators.add(locator)
            self.data.records.append(
                (
                    locator,
                    data_source,
                    width,
                    height,
                    None,
                    None,
                    bounding_box,
                    landmarks,
                    tags or {},
                ),
            )

        @validate_arguments(config=ValidatorConfig)
        def add_augmented(
            self,
            original_locator: str,
            augmented_locator: str,
            augmentation_spec: Dict[str, Any],
            width: Optional[int] = None,  # if absent, original width is used
            height: Optional[int] = None,  # if absent, original height is used
            bounding_box: Optional[np.ndarray] = None,  # if absent, original bbox is used if defined
            landmarks: Optional[np.ndarray] = None,  # if absent, original lmks are used if defined
            tags: Optional[Dict[str, str]] = None,  # note that tags are not propagated forward from the original
        ) -> None:
            """
            Add an augmented version of an existing image to Kolena.

            Note that the original image must already be registered in a previous pass. Tags on the original image are
            not propagated forward to the augmented image.

            :param original_locator: the bucket locator for the original version of this image within the platform
            :param augmented_locator: the bucket locator for the augmented image being registered
            :param augmentation_spec: free-form JSON specification for the augmentation applied to this image
            :param width: optionally specify the width of the augmented image. When absent, the width of the
                corresponding original image is used
            :param height: optionally specify the height of the augmented image. When absent, the height of the
                corresponding original image is used
            :param bounding_box: optionally specify a new bounding box for the augmented image. When absent, any
                bounding box corresponding to the original image is used
            :param landmarks: optionally specify a new set of landmarks for the augmented image. When absent, any set of
                landmarks corresponding to the original image is used
            :param tags: optionally specify a set of tags to associate with the augmented image
            """
            if augmented_locator in self.data.locators:
                raise ValueError(f"duplicate locator: {augmented_locator}")
            self.data.locators.add(augmented_locator)
            self.data.records.append(
                (
                    augmented_locator,
                    None,
                    width or -1,
                    height or -1,
                    original_locator,
                    augmentation_spec,
                    bounding_box,
                    landmarks,
                    tags or {},
                ),
            )

    @classmethod
    @contextmanager
    def register(cls) -> Iterator[Registrar]:
        """
        Context-managed interface to register new images with Kolena. Images with locators that already exist in the
        platform will have their metadata updated. All changes are committed when the context is exited.

        :raises RemoteError: if the registered images were unable to be successfully committed for any reason
        """
        log.info("registering test images")
        registrar = TestImages.Registrar.__factory__(TestImages._Registrar(records=[], locators=set()))
        yield registrar

        init_response = init_upload()
        df = pd.DataFrame(registrar.data.records, columns=TEST_IMAGE_COLUMNS)
        df["image_id"] = -1
        df_validated = TestImageDataFrame(validate_df_schema(df, TestImageDataFrameSchema))
        df_serializable = df_validated.as_serializable()

        upload_data_frame(df=df_serializable, batch_size=BatchSize.UPLOAD_RECORDS.value, load_uuid=init_response.uuid)
        request = LoadAPI.WithLoadUUID(uuid=init_response.uuid)
        finalize_res = krequests.put(
            endpoint_path=API.Path.COMPLETE_REGISTER.value,
            data=json.dumps(dataclasses.asdict(request)),
        )
        krequests.raise_for_status(finalize_res)
        log.success("registered test images")

    @classmethod
    @validate_arguments(config=ValidatorConfig)
    def iter(
        cls,
        data_source: Optional[Union[str, TestSuite, TestSuite.Data, TestCase, TestCase.Data]] = None,
        include_augmented: bool = False,
        batch_size: int = 10_000_000,
    ) -> Iterator[TestImageDataFrame]:
        """
        Iterator of DataFrames describing images registered in the Kolena platform.

        :param data_source: optionally specify the single data source to be retrieved, e.g. ``"my-data-source"``.
            Alternatively, provide a :class:`kolena.fr.TestSuite` or :class:`kolena.fr.TestCase` as source.
            If no argument is provided, all images registered using :meth:`TestImages.register` are returned
        :param include_augmented: optionally specify that augmented images should be returned. By default, only
            original images are returned. Ignored when test case or test suite is provided as ``data_source``
        :param batch_size: optionally specify maximum number of rows to be returned in a single DataFrame. By default,
            limits row count to 10_000_000.
        """
        test_suite_data = data_source.data if isinstance(data_source, TestSuite) else data_source
        test_suite_id = test_suite_data.id if isinstance(test_suite_data, TestSuite.Data) else None
        test_case_data = data_source.data if isinstance(data_source, TestCase) else data_source
        test_case_id = test_case_data.id if isinstance(test_case_data, TestCase.Data) else None
        data_source_display_name = cls._data_source_display_name(data_source, include_augmented)
        from_extra = f" from '{data_source_display_name}'" if data_source_display_name is not None else ""
        log.info(f"loading test images{from_extra}")
        init_request = API.InitLoadRequest(
            include_augmented=include_augmented,
            data_source=data_source if isinstance(data_source, str) else None,
            test_suite_id=test_suite_id,
            test_case_id=test_case_id,
            batch_size=batch_size,
        )
        yield from _BatchedLoader.iter_data(
            init_request=init_request,
            endpoint_path=API.Path.INIT_LOAD_REQUEST.value,
            df_class=TestImageDataFrame,
        )
        log.info(f"loaded test images{from_extra}")

    @staticmethod
    def _data_source_display_name(
        data_source: Optional[Union[str, TestSuite, TestSuite.Data, TestCase, TestCase.Data]],
        include_augmented: bool,
    ) -> Optional[str]:
        if isinstance(data_source, str):
            augmented = " (including augmented images)" if include_augmented else ""
            return f"data source '{data_source}'{augmented}"
        if isinstance(data_source, (TestCase, TestCase.Data)):
            return f"test case '{data_source.data.name if isinstance(data_source, TestCase) else data_source.name}'"
        if isinstance(data_source, (TestSuite, TestSuite.Data)):
            return f"test suite '{data_source.data.name if isinstance(data_source, TestSuite) else data_source.name}'"
        return None
