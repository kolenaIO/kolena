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
from abc import abstractmethod
from collections import OrderedDict
from types import TracebackType
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import Union

import pandera as pa
from pydantic import validate_arguments

from kolena._api.v1.detection import CustomMetrics
from kolena._api.v1.detection import Metrics
from kolena._api.v1.detection import TestRun as API
from kolena._utils import krequests
from kolena._utils import log
from kolena._utils.batched_load import _BatchedLoader
from kolena._utils.batched_load import DFType
from kolena._utils.batched_load import init_upload
from kolena._utils.batched_load import upload_data_frame_chunk
from kolena._utils.consts import BatchSize
from kolena._utils.datatypes import LoadableDataFrame
from kolena._utils.frozen import Frozen
from kolena._utils.instrumentation import report_crash
from kolena._utils.instrumentation import WithTelemetry
from kolena._utils.serde import from_dict
from kolena._utils.validators import ValidatorConfig
from kolena.detection._internal import BaseModel
from kolena.detection._internal import BaseTestImage
from kolena.detection._internal import BaseTestSuite
from kolena.detection._internal import Inference
from kolena.detection._internal.model import SampleInferences
from kolena.errors import CustomMetricsException
from kolena.errors import IncorrectUsageError
from kolena.errors import InputValidationError
from kolena.errors import WorkflowMismatchError

_ImageDataFrame = Union[pa.typing.DataFrame, LoadableDataFrame]
InferenceType = TypeVar("InferenceType")
CustomMetricsCallback = Callable[[List[SampleInferences]], CustomMetrics]


class BaseTestRun(ABC, Frozen, WithTelemetry):
    """
    Base interface to run tests

    :param model: the model being tested.
    :param test_suite: the test suite on which to test the model.
    """

    _TestImageClass: Type[BaseTestImage] = BaseTestImage
    _InferenceClass: Type[InferenceType] = InferenceType
    _ImageResultDataFrameClass: Type[_ImageDataFrame] = _ImageDataFrame
    _LoadTestImagesDataFrameClass: Type[DFType] = DFType

    @validate_arguments(config=ValidatorConfig)
    def __init__(
        self,
        model: BaseModel,
        test_suite: BaseTestSuite,
        config: Optional[Metrics.RunConfig] = None,
        custom_metrics_callback: Optional[CustomMetricsCallback[_TestImageClass, _InferenceClass]] = None,
        reset: bool = False,
    ):
        if model._workflow != test_suite._workflow:
            raise WorkflowMismatchError(
                f"mismatching test suite workflow for model of type '{model._workflow}': '{test_suite._workflow}'",
            )

        if reset:
            log.warn("overwriting existing inferences from this model (reset=True)")
        else:
            log.info("not overwriting any existing inferences from this model (reset=False)")

        request = API.CreateOrRetrieveRequest(
            model_id=model._id,
            test_suite_ids=[test_suite._id],
            config=config,
        )
        res = krequests.post(
            endpoint_path=API.Path.CREATE_OR_RETRIEVE.value,
            data=json.dumps(dataclasses.asdict(request)),
        )
        krequests.raise_for_status(res)
        response = from_dict(data_class=API.CreateOrRetrieveResponse, data=res.json())
        self._id = response.test_run_id
        self._model = model
        self._test_suite = test_suite
        self._locator_to_image_id: Dict[str, int] = {}
        self._inferences: Dict[int, List[Optional[Inference]]] = OrderedDict()
        self._ignored_image_ids: List[int] = []
        self._upload_uuid: Optional[str] = None
        self._n_inferences = 0
        self._custom_metrics_callback: CustomMetricsCallback = custom_metrics_callback
        self._active = False
        self._reset = reset
        # note not calling self._freeze()

    def __enter__(self) -> "BaseTestRun":
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self._upload_chunk()
        self._finalize_upload()
        self._submit_custom_metrics()
        self._active = False
        if exc_type is not None:
            report_crash(self._id, API.Path.MARK_CRASHED.value)

    @validate_arguments(config=ValidatorConfig)
    def add_inferences(self, image: _TestImageClass, inferences: Optional[List[_InferenceClass]]) -> None:
        """
        Adds inferences for a test image to the test run results.

        :param image: the image that inferences are evaluated on
        :param inferences: list of inferences corresponding to the image
        """
        self._assert_active()

        if image.locator not in self._locator_to_image_id.keys():
            raise InputValidationError(
                f"Unrecognized locator '{image.locator}'. Images must be loaded and processed in the same context",
            )
        image_id = self._locator_to_image_id[image.locator]
        if inferences is None:
            self._ignored_image_ids.append(image_id)
            self._n_inferences += 1

        else:
            context_image_inferences = self._inferences.get(image_id, [])
            if len(inferences) == 0:
                context_image_inferences.append(None)
                self._n_inferences += 1
            else:
                context_image_inferences.extend(inferences)
                self._n_inferences += len(inferences)

            self._inferences[image_id] = context_image_inferences

        if self._n_inferences >= BatchSize.UPLOAD_RESULTS.value:
            self._upload_chunk()

    @validate_arguments(config=ValidatorConfig)
    def iter_images(self) -> Iterator[_TestImageClass]:
        """
        Returns an iterator of all remaining images that need inferences evaluated.
        """
        self._assert_active()
        for df_image_batch in self._iter_image_batch():
            for record in df_image_batch.itertuples():
                yield self._image_from_load_image_record(record)

    @validate_arguments(config=ValidatorConfig)
    def load_images(self, batch_size: int = BatchSize.LOAD_SAMPLES.value) -> List[_TestImageClass]:
        """
        Returns a list of images that still need inferences evaluated, bounded in count
        by batch_size. Note that image ground truths will be excluded from the returned
        batch of images.

        :param batch_size: the maximum number of images to retrieve
        """
        self._assert_active()
        log.info("loading batch of images for test run")
        try:
            df_image_batch = next(self._iter_image_batch(batch_size=batch_size))
        except StopIteration:
            # no more images
            return []
        log.info("loaded batch of images for test run")
        return [self._image_from_load_image_record(record) for record in df_image_batch.itertuples()]

    @validate_arguments(config=ValidatorConfig)
    def _iter_image_batch(
        self,
        batch_size: int = BatchSize.LOAD_SAMPLES.value,
    ) -> Iterator[_LoadTestImagesDataFrameClass]:
        if batch_size <= 0:
            raise InputValidationError(f"invalid batch_size '{batch_size}': expected positive integer")
        init_request = API.InitLoadRemainingImagesRequest(
            test_run_id=self._id,
            batch_size=batch_size,
            load_all=self._reset,
        )
        yield from _BatchedLoader.iter_data(
            init_request=init_request,
            endpoint_path=API.Path.INIT_LOAD_REMAINING_IMAGES.value,
            df_class=self._LoadTestImagesDataFrameClass,
        )

    @validate_arguments(config=ValidatorConfig)
    def _upload_chunk(self) -> None:
        if self._n_inferences == 0:
            # Bail if this happens to being run by fencepost immediately after being run by add_inference
            return

        log.info(f"uploading {self._n_inferences} inferences for test run")
        if self._upload_uuid is None:
            init_response = init_upload()
            self._upload_uuid = init_response.uuid

        df_chunk = self._ImageResultDataFrameClass.from_image_inference_mapping(
            self._id,
            self._model._id,
            self._inferences,
            self._ignored_image_ids,
        )
        df_chunk_serializable = df_chunk.as_serializable()
        upload_data_frame_chunk(df_chunk_serializable, load_uuid=self._upload_uuid)
        self._n_inferences = 0
        self._inferences = OrderedDict()
        log.success(f"uploaded {self._n_inferences} inferences for test run")

    def _finalize_upload(self) -> None:
        if self._upload_uuid is None:
            # nothing was uploaded
            return

        log.info("finalizing inference upload for test run")
        request = API.UploadImageResultsRequest(uuid=self._upload_uuid, test_run_id=self._id, reset=self._reset)
        finalize_res = krequests.put(
            endpoint_path=API.Path.UPLOAD_IMAGE_RESULTS.value,
            data=json.dumps(dataclasses.asdict(request)),
        )
        krequests.raise_for_status(finalize_res)
        log.success("finalized inference upload for test run")

    def _assert_active(self) -> None:
        if not self._active:
            raise IncorrectUsageError("test run must be used inside a context manager")

    @abstractmethod
    def _image_from_load_image_record(self, record: Any) -> _TestImageClass:
        ...

    @validate_arguments(config=ValidatorConfig)
    def _compute_custom_metrics(self) -> Dict[int, Dict[int, CustomMetrics]]:
        log.info("computing custom metrics for test run")
        test_case_metrics = {}
        custom_metrics = {}  # { test_suite_id: { test_case_id: CustomMetrics } }

        test_suite = self._test_suite
        log.info(f"computing custom metrics for test suite '{test_suite.name}'")
        test_suite_metrics = {}
        all_inferences = self._model.load_inferences_by_test_case(test_suite)
        for test_case in log.progress_bar(test_suite.test_cases):
            test_case_id = test_case._id
            if test_case_id not in test_case_metrics:
                inferences = all_inferences.get(test_case_id, [])
                try:
                    test_case_metrics[test_case_id] = self._custom_metrics_callback(inferences)
                except Exception as e:
                    raise CustomMetricsException(
                        f"Error encountered computing custom metrics for test case '{test_case.name}'",
                    ) from e

            test_suite_metrics[test_case_id] = test_case_metrics[test_case_id]

        custom_metrics[test_suite._id] = test_suite_metrics
        log.success(f"computed custom metrics for test suite '{test_suite.name}'")

        log.success("computed custom metrics for test run")
        return custom_metrics

    def _submit_custom_metrics(self) -> None:
        if self._custom_metrics_callback is None:
            return

        log.info("computing and uploading custom metrics for test run")
        custom_metrics = self._compute_custom_metrics()
        request = API.UpdateCustomMetricsRequest(model_id=self._model._id, metrics=custom_metrics)
        res = krequests.put(
            endpoint_path=API.Path.UPLOAD_CUSTOM_METRICS.value,
            data=json.dumps(dataclasses.asdict(request)),
        )
        krequests.raise_for_status(res)
        log.success("computed and uploaded custom metrics for test run")
