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
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

from pydantic import validate_arguments

from kolena._utils import log
from kolena._utils.validators import ValidatorConfig
from kolena.classification import InferenceModel
from kolena.classification import Model
from kolena.classification import TestConfig
from kolena.classification import TestImage
from kolena.classification import TestSuite
from kolena.classification.test_config import AccuracyOptimal
from kolena.detection._datatypes import ImageResultDataFrame
from kolena.detection._internal import BaseTestRun
from kolena.detection._internal.datatypes import LoadTestImagesDataFrame
from kolena.detection._internal.metadata import _from_dict
from kolena.detection._internal.test_run import CustomMetricsCallback
from kolena.detection.inference import ClassificationLabel


class TestRun(BaseTestRun):
    """
    Interface to run tests for a :class:`kolena.classification.Model` on a set of
    :class:`kolena.classification.TestSuite` suites. Any in-progress tests for this model on these suites are resumed.

    For a streamlined interface, see :meth:`kolena.classification.test`.

    :param model: the model being tested.
    :param test_suite: the test suite on which to test the model.
    :param test_config: Optionally specify a :class:`kolena.classification.TestConfig` to customize the metrics
                           evaluation logic for this test run.
                           Defaults to :class:`kolena.classification.test_config.AccuracyOptimal` if unspecified.
    :param custom_metrics_callback: Optionally specify a callback function to compute custom metrics for each test-case.
                                        The callback would be passed inferences of images in each testcase and should
                                        return a dictionary with metric name as key and metric value as value.
    :param reset: overwrites existing inferences if set.
    """

    _TestImageClass = TestImage
    _InferenceClass = Tuple[str, float]
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
        config = AccuracyOptimal() if test_config is None else test_config
        super().__init__(
            model,
            test_suite,
            config=config._to_run_config(),
            custom_metrics_callback=custom_metrics_callback,
            reset=reset,
        )

    @validate_arguments(config=ValidatorConfig)
    def add_inferences(self, image: _TestImageClass, inferences: Optional[List[_InferenceClass]]) -> None:
        super().add_inferences(
            image,
            [ClassificationLabel(label, confidence) for label, confidence in inferences]
            if inferences is not None
            else None,
        )

    def _image_from_load_image_record(self, record: Any) -> _TestImageClass:
        self._locator_to_image_id[record.locator] = record.test_sample_id
        return self._TestImageClass(
            locator=record.locator,
            dataset=record.dataset,
            metadata=_from_dict(record.metadata),
            labels=[],  # note that no ground truth labels are loaded during testing
        )


@validate_arguments(config=ValidatorConfig)
def test(
    model: InferenceModel,
    test_suite: TestSuite,
    test_config: Optional[TestConfig] = None,
    custom_metrics_callback: Optional[CustomMetricsCallback[TestImage, Tuple[str, float]]] = None,
    reset: bool = False,
) -> None:
    """
    Test the provided :class:`kolena.classification.InferenceModel`` on one or more provided
    :class:`kolena.detection.TestSuite` suites. Any tests already in progress for this model on these suites are
    resumed.

    :param model: the model being tested, complete with :meth:`kolena.classification.InferenceModel.infer` function
        to perform inference.
    :param test_suite: the test suite on which to test the model.
    :param test_config: Optionally specify a :class:`kolena.classification.TestConfig` to customize the metrics
                           evaluation logic for this test run.
                           Defaults to :class:`kolena.classification.test_config.AccuracyOptimal` if unspecified.
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
