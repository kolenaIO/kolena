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
import dataclasses
import json
import time
from abc import ABCMeta
from collections import defaultdict
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import pandas as pd

from kolena._api.v1.event import EventAPI
from kolena._api.v1.generic import TestRun as API
from kolena._utils import krequests
from kolena._utils import log
from kolena._utils.batched_load import _BatchedLoader
from kolena._utils.batched_load import init_upload
from kolena._utils.batched_load import upload_data_frame
from kolena._utils.consts import BatchSize
from kolena._utils.dataframes.validators import validate_df_schema
from kolena._utils.endpoints import get_results_url
from kolena._utils.frozen import Frozen
from kolena._utils.instrumentation import report_crash
from kolena._utils.instrumentation import with_event
from kolena._utils.pydantic_v1 import validate_arguments
from kolena._utils.serde import from_dict
from kolena._utils.validators import ValidatorConfig
from kolena.errors import IncorrectUsageError
from kolena.errors import InputValidationError
from kolena.errors import WorkflowMismatchError
from kolena.workflow import Evaluator
from kolena.workflow import EvaluatorConfiguration
from kolena.workflow import GroundTruth
from kolena.workflow import Inference
from kolena.workflow import MetricsTestCase
from kolena.workflow import MetricsTestSample
from kolena.workflow import MetricsTestSuite
from kolena.workflow import Model
from kolena.workflow import Plot
from kolena.workflow import TestCase
from kolena.workflow import TestSample
from kolena.workflow import TestSuite
from kolena.workflow._datatypes import MetricsDataFrame
from kolena.workflow._datatypes import MetricsDataFrameSchema
from kolena.workflow._datatypes import TestSampleDataFrame
from kolena.workflow._datatypes import TestSampleDataFrameSchema
from kolena.workflow.evaluator import _configuration_description
from kolena.workflow.evaluator import _maybe_display_name
from kolena.workflow.evaluator import _maybe_evaluator_configuration_to_api
from kolena.workflow.evaluator_function import _is_configured
from kolena.workflow.evaluator_function import _TestCases
from kolena.workflow.evaluator_function import BasicEvaluatorFunction
from kolena.workflow.evaluator_function import EvaluationResults
from kolena.workflow.test_sample import _METADATA_KEY


class TestRun(Frozen, metaclass=ABCMeta):
    """
    A [`Model`][kolena.workflow.Model] tested on a [`TestSuite`][kolena.workflow.TestSuite] using a specific
    [`Evaluator`][kolena.workflow.Evaluator] implementation.

    :param model: The model being tested.
    :param test_suite: The test suite on which to test the model.
    :param evaluator: An optional evaluator implementation.
        Requires a previously configured server-side evaluator to default to if omitted.
        (Please see [`BasicEvaluatorFunction`][kolena.workflow.evaluator_function.BasicEvaluatorFunction] for type
        definition.)
    :param configurations: a list of configurations to use when running the evaluator.
    :param reset: overwrites existing inferences if set.
    """

    _id: int

    model: Model
    test_suite: TestSuite
    evaluator: Optional[Union[Evaluator, BasicEvaluatorFunction]]
    configurations: Optional[List[EvaluatorConfiguration]]

    @validate_arguments(config=ValidatorConfig)
    def __init__(
        self,
        model: Model,
        test_suite: TestSuite,
        evaluator: Optional[Union[Evaluator, BasicEvaluatorFunction]] = None,
        configurations: Optional[List[EvaluatorConfiguration]] = None,
        reset: bool = False,
    ):
        if configurations is None:
            configurations = []

        is_evaluator_function = evaluator is not None and not isinstance(evaluator, Evaluator)
        if is_evaluator_function and _is_configured(evaluator) and len(configurations) == 0:  # type: ignore
            raise ValueError("evaluator requires configuration but no configurations provided")

        if model.workflow != test_suite.workflow:
            raise WorkflowMismatchError(
                f"model workflow ({model.workflow}) does not match test suite workflow ({test_suite.workflow})",
            )

        if reset:
            log.warn("overwriting existing inferences from this model (reset=True)")
        else:
            log.info("not overwriting any existing inferences from this model (reset=False)")

        self.model = model
        self.test_suite = test_suite
        self.evaluator = evaluator
        self.configurations = evaluator.configurations if isinstance(evaluator, Evaluator) else configurations
        self.reset = reset

        evaluator_display_name = (
            None
            if evaluator is None
            else evaluator.display_name()
            if isinstance(evaluator, Evaluator)
            else getattr(evaluator, "__name__", None)
        )
        api_configurations = (
            [_maybe_evaluator_configuration_to_api(config) for config in self.configurations]
            if self.configurations is not None
            else None
        )

        request = API.CreateOrRetrieveRequest(
            model_id=model._id,
            test_suite_id=test_suite._id,
            evaluator=evaluator_display_name,
            configurations=api_configurations,  # type: ignore
        )
        res = krequests.put(
            endpoint_path=API.Path.CREATE_OR_RETRIEVE.value,
            data=json.dumps(dataclasses.asdict(request)),
        )
        krequests.raise_for_status(res)
        response = from_dict(data_class=API.CreateOrRetrieveResponse, data=res.json())
        self._id = response.test_run_id
        self._freeze()

    @with_event(event_name=EventAPI.Event.EXECUTE_TEST_RUN)
    def run(self) -> None:
        """
        Run the testing process, first extracting inferences for all test samples in the test suite then performing
        evaluation.
        """
        try:
            inferences = []
            for ts in log.progress_bar(self.iter_test_samples(), desc="performing inference"):
                if self.model.infer is None:  # only fail when `infer` is necessary
                    raise ValueError("model must implement `infer`")
                inferences.append((ts, self.model.infer(ts)))

            if len(inferences) > 0:
                log.success(f"performed inference on {len(inferences)} test samples")
                log.info("uploading inferences")
                self.upload_inferences(inferences)

            self.evaluate()
        except Exception as e:
            report_crash(self._id, API.Path.MARK_CRASHED.value)
            raise e

    def load_test_samples(self) -> List[TestSample]:
        """
        Load the test samples in the test suite that do not yet have inferences uploaded.

        :return: a list of all test samples in the test suite still requiring inferences.
        """
        return list(self.iter_test_samples())

    def iter_test_samples(self) -> Iterator[TestSample]:
        """
        Iterate through the test samples in the test suite that do not yet have inferences uploaded.

        :return: an iterator over each test sample still requiring inferences.
        """
        test_sample_type = self.model.workflow.test_sample_type
        for df_batch in self._iter_test_samples_batch():
            for record in df_batch.itertuples():
                test_sample = test_sample_type._from_dict(
                    {**record.test_sample, _METADATA_KEY: record.test_sample_metadata},
                )
                yield test_sample

    def _iter_all_inferences(self) -> Iterator[Tuple[TestSample, GroundTruth, Inference]]:
        """
        Iterate over all inferences stored for this model on the provided test case.

        :return: an iterator exposing the ground truths and inferences for all test samples in the test run.
        """
        log.info(f"loading inferences from model '{self.model.name}' on test suite '{self.test_suite.name}'")
        for df_batch in _BatchedLoader.iter_data(
            init_request=API.LoadTestSampleInferencesRequest(
                test_run_id=self._id,
                batch_size=BatchSize.LOAD_SAMPLES.value,
            ),
            endpoint_path=API.Path.LOAD_INFERENCES.value,
            df_class=TestSampleDataFrame,
        ):
            for record in df_batch.itertuples():
                test_sample = self.test_suite.workflow.test_sample_type._from_dict(
                    {**record.test_sample, _METADATA_KEY: record.test_sample_metadata},
                )
                ground_truth = self.test_suite.workflow.ground_truth_type._from_dict(record.ground_truth)
                inference = self.test_suite.workflow.inference_type._from_dict(record.inference)
                yield test_sample, ground_truth, inference
        log.info(f"loaded inferences from model '{self.model.name}' on test suite '{self.test_suite.name}'")

    @validate_arguments(config=ValidatorConfig)
    def upload_inferences(self, inferences: List[Tuple[TestSample, Inference]]) -> None:
        """
        Upload inferences from a model.

        :param inferences: the inferences, paired with their corresponding test samples, to upload.
        """
        if len(inferences) == 0:
            return

        inference_dicts = [(ts._to_dict(), inf._to_dict()) for ts, inf in inferences]
        df = pd.DataFrame(inference_dicts, columns=["test_sample", "inference"])
        df_validated = TestSampleDataFrame(validate_df_schema(df, TestSampleDataFrameSchema, trusted=True))
        df_serializable = df_validated.as_serializable()

        init_response = init_upload()
        upload_data_frame(df_serializable, init_response.uuid)
        request = API.UploadInferencesRequest(uuid=init_response.uuid, test_run_id=self._id, reset=self.reset)
        res = krequests.put(
            endpoint_path=API.Path.UPLOAD_INFERENCES.value,
            data=json.dumps(dataclasses.asdict(request)),
        )
        krequests.raise_for_status(res)

    def evaluate(self) -> None:
        """
        Perform evaluation by computing metrics for individual test samples, in aggregate across test cases, and across
        the complete test suite at each [`EvaluatorConfiguration`][kolena.workflow.EvaluatorConfiguration].
        """
        if self.evaluator is None:
            log.info("commencing server side metrics evaluation")
            self._start_server_side_evaluation()
            return

        # TODO: assert that testing is complete?
        t0 = time.time()
        log.info("commencing evaluation")
        if isinstance(self.evaluator, Evaluator):
            self._perform_evaluation(self.evaluator)
        else:
            self._perform_streamlined_evaluation(self.evaluator)  # type: ignore

        log.success(f"completed evaluation in {time.time() - t0:0.1f} seconds")
        log.success(f"results: {get_results_url(self.model.workflow.name, self.model._id, self.test_suite._id)}")

    def _perform_evaluation(self, evaluator: Evaluator) -> None:
        configurations: Sequence[Optional[EvaluatorConfiguration]] = (
            cast(Sequence[Optional[EvaluatorConfiguration]], evaluator.configurations)
            if len(evaluator.configurations) > 0
            else [None]
        )
        test_case_metrics: Dict[int, Dict[Optional[EvaluatorConfiguration], MetricsTestCase]] = {}
        test_case_plots: Dict[int, Dict[Optional[EvaluatorConfiguration], Optional[List[Plot]]]] = {}

        for test_case in self.test_suite.test_cases:
            log.info(f"evaluating test case '{test_case.name}'")
            test_case_metrics_by_config = {}
            test_case_plots_by_config = {}
            inferences: List[Tuple[TestSample, GroundTruth, Inference]] = self.model.load_inferences(test_case)

            for configuration in configurations:
                configuration_description = _configuration_description(configuration)
                log.info(f"computing test sample metrics {configuration_description}")
                metrics_test_sample = evaluator.compute_test_sample_metrics(test_case, inferences, configuration)

                log.info(f"uploading test sample metrics {configuration_description}")
                self._upload_test_sample_metrics(
                    test_case,
                    metrics_test_sample,
                    configuration,
                    self.model._id,
                )

                log.info(f"computing test case metrics {configuration_description}")
                # TODO: sort? order returned from evaluator may not match inferences order
                mts = [metrics for _, metrics in metrics_test_sample]
                metrics_test_case = evaluator.compute_test_case_metrics(test_case, inferences, mts, configuration)
                test_case_metrics_by_config[configuration] = metrics_test_case

                log.info(f"computing test case plots {configuration_description}")
                plots_test_case = evaluator.compute_test_case_plots(test_case, inferences, mts, configuration)
                test_case_plots_by_config[configuration] = plots_test_case

            test_case_metrics[test_case._id] = test_case_metrics_by_config
            test_case_plots[test_case._id] = test_case_plots_by_config

        log.info("uploading test case metrics")
        self._upload_test_case_metrics(test_case_metrics)
        log.info("uploading test case plots")
        self._upload_test_case_plots(test_case_plots)

        log.info("computing test suite metrics")
        test_suite_metrics: Dict[Optional[EvaluatorConfiguration], Optional[MetricsTestSuite]] = {}
        for configuration in configurations:
            test_case_with_metrics = [
                (tc, test_case_metrics[tc._id][configuration]) for tc in self.test_suite.test_cases
            ]
            log.info(f"computing test suite metrics {_configuration_description(configuration)}")
            metrics_test_suite = evaluator.compute_test_suite_metrics(
                self.test_suite,
                test_case_with_metrics,
                configuration,
            )
            test_suite_metrics[configuration] = metrics_test_suite

        log.info("uploading test suite metrics")
        self._upload_test_suite_metrics(test_suite_metrics)

    def _perform_streamlined_evaluation(self, evaluator: BasicEvaluatorFunction) -> None:
        test_samples, ground_truths, inferences = [], [], []
        for sample, gt, inf in self._iter_all_inferences():
            test_samples.append(sample)
            ground_truths.append(gt)
            inferences.append(inf)
        test_case_membership: List[Tuple[TestCase, List[TestSample]]] = self.test_suite.load_test_samples()
        test_case_test_samples = _TestCases(
            test_case_membership,
            self._id,
            len(self.configurations or []),
        )

        test_case_metrics: Dict[int, Dict[Optional[EvaluatorConfiguration], MetricsTestCase]] = defaultdict(dict)
        test_case_plots: Dict[int, Dict[Optional[EvaluatorConfiguration], List[Plot]]] = defaultdict(dict)
        test_suite_metrics: Dict[Optional[EvaluatorConfiguration], MetricsTestSuite] = dict()

        def process_results(results: Optional[EvaluationResults], config: Optional[EvaluatorConfiguration]) -> None:
            if results is None:
                log.info(f"no results {_configuration_description(config)}")
                return

            log.info(f"uploading test sample metrics {_configuration_description(config)}")
            self._upload_test_sample_metrics(
                test_case=None,
                metrics=results.metrics_test_sample,
                configuration=config,
                model_id=self.model._id,
            )
            for test_case, metrics in results.metrics_test_case:
                test_case_metrics[test_case._id][config] = metrics
            for test_case, plots in results.plots_test_case:
                test_case_plots[test_case._id][config] = plots
            if results.metrics_test_suite is not None:
                test_suite_metrics[config] = results.metrics_test_suite

        if _is_configured(evaluator) and self.configurations:
            for configuration in self.configurations:
                test_case_test_samples._set_configuration(configuration)
                evaluation_results = evaluator(
                    test_samples,
                    ground_truths,
                    inferences,
                    test_case_test_samples,
                    configuration,
                )  # type: ignore
                process_results(evaluation_results, configuration)
        else:
            test_case_test_samples._set_configuration(None)
            evaluation_results = evaluator(
                test_samples,
                ground_truths,
                inferences,
                test_case_test_samples,
            )  # type: ignore
            process_results(evaluation_results, None)

        log.info("uploading test case metrics")
        self._upload_test_case_metrics(test_case_metrics)
        log.info("uploading test case plots")
        self._upload_test_case_plots(test_case_plots)
        log.info("uploading test suite metrics")
        self._upload_test_suite_metrics(test_suite_metrics)

    def _iter_test_samples_batch(
        self,
        batch_size: int = BatchSize.LOAD_SAMPLES.value,
    ) -> Iterator[TestSampleDataFrame]:
        if batch_size <= 0:
            raise InputValidationError(f"invalid batch_size '{batch_size}': expected positive integer")
        init_request = API.LoadRemainingTestSamplesRequest(
            test_run_id=self._id,
            batch_size=batch_size,
            load_all=self.reset,
        )
        yield from _BatchedLoader.iter_data(
            init_request=init_request,
            endpoint_path=API.Path.LOAD_TEST_SAMPLES.value,
            df_class=TestSampleDataFrame,
        )

    def _extract_thresholded_metrics(
        self,
        records: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    ) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any]]], List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]]:
        standard_metrics = []
        thresholded_metrics: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []

        for first_record, second_record in records:
            keys_to_remove = []
            for key, value in second_record.items():
                if isinstance(value, list):
                    thresholded_metrics.extend(
                        (first_record, dict(name=key, **item))
                        for item in value
                        if isinstance(item, dict) and item.get("data_type") == "METRICS/THRESHOLDED"
                    )

                    if any(isinstance(item, dict) and item.get("data_type") == "METRICS/THRESHOLDED" for item in value):
                        keys_to_remove.append(key)

            for key in keys_to_remove:
                second_record.pop(key, None)

            standard_metrics.append((first_record, second_record))

        return standard_metrics, thresholded_metrics

    @validate_arguments(config=ValidatorConfig)
    def _upload_test_sample_metrics(
        self,
        test_case: Optional[TestCase],
        metrics: List[Tuple[TestSample, MetricsTestSample]],
        configuration: Optional[EvaluatorConfiguration],
        model_id: int,
    ) -> None:
        metrics_records = [(ts._to_dict(), ts_metrics._to_dict()) for ts, ts_metrics in metrics]
        metrics_records, thresholded_metrics = self._extract_thresholded_metrics(metrics_records)
        log.info(f"uploading {len(metrics_records)} test sample metrics")
        df = pd.DataFrame(metrics_records, columns=["test_sample", "metrics"])
        df_validated = MetricsDataFrame(validate_df_schema(df, MetricsDataFrameSchema, trusted=True))
        df_serializable = df_validated.as_serializable()

        init_response = init_upload()
        upload_data_frame(df_serializable, init_response.uuid)

        request = API.UploadTestSampleMetricsRequest(
            uuid=init_response.uuid,
            test_run_id=self._id,
            test_case_id=test_case._id if test_case is not None else None,
            configuration=_maybe_evaluator_configuration_to_api(configuration),
        )
        res = krequests.put(
            endpoint_path=API.Path.UPLOAD_TEST_SAMPLE_METRICS.value,
            data=json.dumps(dataclasses.asdict(request)),
        )
        krequests.raise_for_status(res)

        if len(thresholded_metrics) > 0:
            df = pd.DataFrame(thresholded_metrics, columns=["test_sample", "metrics"])
            df_validated = MetricsDataFrame(validate_df_schema(df, MetricsDataFrameSchema, trusted=True))
            df_serializable = df_validated.as_serializable()
            init_response = init_upload()
            upload_data_frame(df_serializable, init_response.uuid)

            thresholded_metrics_request = API.UploadTestSampleThresholdedMetricsRequest(
                uuid=init_response.uuid,
                test_run_id=self._id,
                test_case_id=test_case._id if test_case is not None else None,
                configuration=_maybe_evaluator_configuration_to_api(configuration),
                model_id=model_id,
            )
            res = krequests.put(
                endpoint_path=API.Path.UPLOAD_TEST_SAMPLE_METRICS_THRESHOLDED.value,
                data=json.dumps(dataclasses.asdict(thresholded_metrics_request)),
            )
            krequests.raise_for_status(res)

    def _upload_test_case_metrics(
        self,
        metrics: Dict[int, Dict[Optional[EvaluatorConfiguration], MetricsTestCase]],
    ) -> None:
        records = [
            (test_case_id, _maybe_display_name(config), tc_metrics._to_dict())
            for test_case_id, tc_metrics_by_config in metrics.items()
            for config, tc_metrics in tc_metrics_by_config.items()
        ]
        df = pd.DataFrame(records, columns=["test_case_id", "configuration_display_name", "metrics"])
        return self._upload_aggregate_metrics(API.Path.UPLOAD_TEST_CASE_METRICS.value, df)

    def _upload_test_case_plots(
        self,
        plots: Mapping[int, Mapping[Optional[EvaluatorConfiguration], Optional[List[Plot]]]],
    ) -> None:
        records = [
            (test_case_id, _maybe_display_name(config), tc_plot._to_dict())
            for test_case_id, tc_plots_by_config in plots.items()
            for config, tc_plots in tc_plots_by_config.items()
            for tc_plot in tc_plots or []
        ]
        df = pd.DataFrame(records, columns=["test_case_id", "configuration_display_name", "metrics"])
        return self._upload_aggregate_metrics(API.Path.UPLOAD_TEST_CASE_PLOTS.value, df)

    def _upload_test_suite_metrics(
        self,
        metrics: Mapping[Optional[EvaluatorConfiguration], Optional[MetricsTestSuite]],
    ) -> None:
        records: List[Tuple[Optional[str], Dict[str, Any]]] = [
            (_maybe_display_name(config), ts_metrics._to_dict())
            for config, ts_metrics in metrics.items()
            if ts_metrics is not None
        ]
        df = pd.DataFrame(records, columns=["configuration_display_name", "metrics"])
        return self._upload_aggregate_metrics(API.Path.UPLOAD_TEST_SUITE_METRICS.value, df)

    def _upload_aggregate_metrics(self, endpoint_path: str, df: pd.DataFrame) -> None:
        df_validated = MetricsDataFrame(validate_df_schema(df, MetricsDataFrameSchema, trusted=True))
        df_serializable = df_validated.as_serializable()

        init_response = init_upload()
        upload_data_frame(df_serializable, init_response.uuid)

        request = API.UploadAggregateMetricsRequest(
            uuid=init_response.uuid,
            test_run_id=self._id,
            test_suite_id=self.test_suite._id,
        )
        res = krequests.put(endpoint_path=endpoint_path, data=json.dumps(dataclasses.asdict(request)))
        krequests.raise_for_status(res)

    def _start_server_side_evaluation(self) -> None:
        request = API.EvaluateRequest(test_run_id=self._id)
        res = krequests.put(endpoint_path=API.Path.EVALUATE.value, data=json.dumps(dataclasses.asdict(request)))
        krequests.raise_for_status(res)


@validate_arguments(config=ValidatorConfig)
def test(
    model: Model,
    test_suite: TestSuite,
    evaluator: Optional[Union[Evaluator, BasicEvaluatorFunction]] = None,
    configurations: Optional[List[EvaluatorConfiguration]] = None,
    reset: bool = False,
) -> None:
    """
    Test a [`Model`][kolena.workflow.Model] on a [`TestSuite`][kolena.workflow.TestSuite] using a specific
    [`Evaluator`][kolena.workflow.Evaluator] implementation.

    ```python
    from kolena.workflow import test

    test(model, test_suite, evaluator, reset=True)
    ```

    :param model: The model being tested, implementing the `infer` method.
    :param test_suite: The test suite on which to test the model.
    :param evaluator: An optional evaluator implementation.
        Requires a previously configured server-side evaluator to default to if omitted.
        (Please see [`BasicEvaluatorFunction`][kolena.workflow.evaluator_function.BasicEvaluatorFunction] for type
        definition.)
    :param configurations: A list of configurations to use when running the evaluator.
    :param reset: Overwrites existing inferences if set.
    """
    if not test_suite.test_cases:
        raise IncorrectUsageError(
            f"test suite '{test_suite.name}' has no test cases, please add test cases" f" to the test suite",
        )
    TestRun(model, test_suite, evaluator, configurations, reset).run()
