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
import random
import time
import uuid

import pandas as pd
from pandas._testing import assert_frame_equal

from kolena._experimental.trace.trace import KOLENA_DEFAULT_ID
from kolena._experimental.trace.trace import KOLENA_TIME_ELAPSED_KEY
from kolena._experimental.trace.trace import KOLENA_TIMESTAMP_KEY
from kolena._experimental.trace.trace import KolenaTrace
from kolena.dataset import download_dataset
from kolena.dataset import download_results


def test__kolena_trace_provided_id() -> None:
    run_id = uuid.uuid4().hex
    dataset_name = "test_kolena_trace_provided_id" + run_id
    model_name = "test_kolena_trace_provided_id_model" + run_id
    expected_datapoints = []
    expected_results = []

    @KolenaTrace(
        dataset_name=dataset_name,
        model_name=model_name,
        id_fields=["a", "b"],
        record_timestamp=False,
        sync_interval=5,
    )
    def predict(a, b, e=2, params=None):
        time.sleep(random.random())
        return {"sum": a + b + e + random.random(), "str": str(f"received _result {a + b + e}")}

    for i in range(20):
        if i == 19:
            time.sleep(30)  # Making sure the last iteration will trigger an sync
        result = predict(i, b=i + 1, params={"a": i, "b": i + 1, "c": i + 2})
        result["a"] = i
        result["b"] = i + 1
        expected_results.append(result)
        expected_datapoints.append({"a": i, "b": i + 1, "e": 2, "params": {"a": i, "b": i + 1, "c": i + 2}})

    time.sleep(30)  # wait for the datasync to finish

    uploaded_datapoints = download_dataset(dataset_name).sort_values(by=["a"], ignore_index=True).reset_index()
    uploaded_results = (
        download_results(dataset_name, model_name)[1][0].results.sort_values(by=["a"], ignore_index=True).reset_index()
    )
    expected_datapoints_df = pd.DataFrame(expected_datapoints).reset_index()
    expected_results_df = pd.DataFrame(expected_results).reset_index()
    assert_frame_equal(uploaded_datapoints, expected_datapoints_df, check_dtype=False, check_like=True)
    assert_frame_equal(
        uploaded_results[["sum", "str"]],
        expected_results_df[["sum", "str"]],
        check_dtype=False,
        check_like=True,
    )


def test__kolena_trace_with_time_and_default_id() -> None:
    run_id = uuid.uuid4().hex
    dataset_name = "test__kolena_trace_with_time_and_default_id" + run_id
    model_name = "test__kolena_trace_with_time_and_default_id" + run_id
    expected_datapoints = []
    expected_results = []

    @KolenaTrace(dataset_name=dataset_name, model_name=model_name, sync_interval=10)
    def predict(a, b, e=2, params=None):
        time.sleep(random.random())
        return {"sum": a + b + e + random.random(), "str": str(f"received _result {a + b + e}")}

    for i in range(20):
        if i == 19:
            time.sleep(30)  # Making sure the last iteration will trigger an sync
        result = predict(i, b=i + 1, params={"a": i, "b": i + 1, "c": i + 2})
        result["a"] = i
        result["b"] = i + 1
        expected_results.append(result)
        expected_datapoints.append({"a": i, "b": i + 1, "e": 2, "params": {"a": i, "b": i + 1, "c": i + 2}})

    time.sleep(30)  # wait for the datasync to finish

    uploaded_datapoints = download_dataset(dataset_name).sort_values(by=["a"], ignore_index=True).reset_index()
    uploaded_results = (
        download_results(dataset_name, model_name)[1][0]
        .results.sort_values(by=["sum"], ignore_index=True)
        .reset_index()
    )
    expected_datapoints_df = pd.DataFrame(expected_datapoints)
    expected_results_df = pd.DataFrame(expected_results)
    assert KOLENA_DEFAULT_ID in uploaded_datapoints.columns
    assert KOLENA_DEFAULT_ID in uploaded_results.columns
    assert KOLENA_TIMESTAMP_KEY in uploaded_datapoints.columns
    assert KOLENA_TIMESTAMP_KEY in uploaded_results.columns
    assert KOLENA_TIME_ELAPSED_KEY in uploaded_results.columns
    assert_frame_equal(
        uploaded_datapoints[["a", "b", "e", "params"]],
        expected_datapoints_df[["a", "b", "e", "params"]],
        check_dtype=False,
    )
    assert_frame_equal(uploaded_results[["sum", "str"]], expected_results_df[["sum", "str"]], check_dtype=False)
