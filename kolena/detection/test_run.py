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
from typing import Optional
from typing import Type

from pydantic import validate_arguments

from kolena._utils import log
from kolena._utils.validators import ValidatorConfig
from kolena.detection import InferenceModel
from kolena.detection import Model
from kolena.detection import TestConfig
from kolena.detection import TestImage
from kolena.detection import TestSuite
from kolena.detection._datatypes import ImageResultDataFrame
from kolena.detection._internal.datatypes import LoadTestImagesDataFrame
from kolena.detection._internal.metadata import _from_dict
from kolena.detection._internal.test_run import BaseTestRun
from kolena.detection._internal.test_run import CustomMetricsCallback
from kolena.detection.inference import Inference
from kolena.detection.test_config import F1Optimal


class TestRun(BaseTestRun):
    """
    Interface to run tests for a :class:`kolena.detection.Model` on a set of :class:`kolena.detection.TestSuite` suites.

    For a streamlined interface, see :meth:`kolena.detection.test`.

    Changes are committed to the Kolena platform during execution and when the context is exited.

    :param model: the model being tested.
    :param test_suite: the test suite on which to test the model.
    :param test_config: Optionally specify a :class:`kolena.detection.TestConfig` to customize the metrics
                           evaluation logic for this test run. Defaults to :class:`kolena.detection.config.F1Optimal`
                           with an iou_threshold of 0.5 if unspecified.
    :param custom_metrics_callback: Optionally specify a callback function to compute custom metrics for each test-case.
                                        The callback would be passed inferences of images in each testcase and should
                                        return a dictionary with metric name as key and metric value as value.
    :param reset: overwrites existing inferences if set.
    """

    _TestImageClass = TestImage
    _InferenceClass = Inference
    _ImageResultDataFrameClass: Type[ImageResultDataFrame] = ImageResultDataFrame
    _LoadTestImagesDataFrameClass: Type[LoadTestImagesDataFrame] = LoadTestImagesDataFrame

    @validate_arguments(config=ValidatorConfig)
    def __init__(
        self,
        model: Model,
        test_suite: TestSuite,
        test_config: Optional[TestConfig] = None,
        custom_metrics_callback: Optional[CustomMetricsCallback[_TestImageClass, _InferenceClass]] = None,
        reset: bool = False,
    ):
        config = F1Optimal(iou_threshold=0.5) if test_config is None else test_config
        super().__init__(
            model,
            test_suite,
            config=config._to_run_config(),
            custom_metrics_callback=custom_metrics_callback,
            reset=reset,
        )

    def _image_from_load_image_record(self, record: Any) -> _TestImageClass:
        self._locator_to_image_id[record.locator] = record.test_sample_id
        return self._TestImageClass(
            locator=record.locator,
            dataset=record.dataset,
            metadata=_from_dict(record.metadata),
            ground_truths=[],  # note that no ground truths are loaded during testing
        )


@validate_arguments(config=ValidatorConfig)
def test(
    model: InferenceModel,
    test_suite: TestSuite,
    test_config: Optional[TestConfig] = None,
    custom_metrics_callback: Optional[CustomMetricsCallback[TestImage, Inference]] = None,
    reset: bool = False,
) -> None:
    """
    Test the provided :class:`kolena.detection.InferenceModel` on one or more provided
    :class:`kolena.detection.TestSuite` suites. Any tests already in progress for this model on these suites are
    resumed.

    :param model: the model being tested.
    :param test_suite: the test suite on which to test the model.
    :param test_config: Optionally specify a :class:`kolena.detection.TestConfig` to customize the metrics
                           evaluation logic for this test run.
                           Defaults to :class:`kolena.detection.test_config.F1Optimal` with an iou_threshold of 0.5
                           if unspecified.
    :param custom_metrics_callback: Optionally specify a callback function to compute custom metrics for each test-case.
                                        The callback would be passed inferences of images in each testcase and should
                                        return a dictionary with metric name as key and metric value as value.
    :param reset: overwrites existing inferences if set.
    """
    with TestRun(
        model,
        test_suite,
        test_config=test_config,
        custom_metrics_callback=custom_metrics_callback,
        reset=reset,
    ) as test_run:
        log.info("performing inference")
        for image in log.progress_bar(test_run.iter_images()):
            test_run.add_inferences(image, model.infer(image))
        log.success("performed inference")
