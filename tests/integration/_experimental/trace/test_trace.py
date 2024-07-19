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
import threading
import time
import uuid

import pandas as pd
from pandas._testing import assert_frame_equal

from kolena._experimental.trace import kolena_trace
from kolena._experimental.trace.trace import KOLENA_DEFAULT_ID
from kolena._experimental.trace.trace import KOLENA_TIME_ELAPSED_KEY
from kolena._experimental.trace.trace import KOLENA_TIMESTAMP_KEY
from kolena.dataset import download_dataset
from kolena.dataset import download_results


def test__kolena_trace_default_id() -> None:
    run_id = uuid.uuid4().hex
    dataset_name = "test_kolena_trace_provided_id" + run_id
    model_name = "test_kolena_trace_provided_id_model" + run_id
    expected_datapoints = []
    expected_results = []

    @kolena_trace(
        dataset_name=dataset_name,
        model_name=model_name,
        record_timestamp=False,
        sync_interval=5,
    )
    def predict(a, b, e=2, params=None):
        time.sleep(1)
        return {"sum": a + b + e + random.random(), "str": str(f"received _result {a + b + e}")}

    for i in range(20):
        if i == 19:
            time.sleep(30)  # Making sure the last iteration will trigger a sync
        result = predict(i, b=i + 1, params={"a": i, "b": i + 1})
        result["a"] = i
        result["b"] = i + 1
        expected_results.append(result)
        expected_datapoints.append({"a": i, "b": i + 1, "e": 2, "params": {"a": i, "b": i + 1}})

    time.sleep(30)  # wait for the datasync to finish

    uploaded_datapoints = download_dataset(dataset_name).sort_values(by=["a"], ignore_index=True).reset_index()
    uploaded_results = (
        download_results(dataset_name, model_name)[1][0]
        .results.sort_values(by=["sum"], ignore_index=True)
        .reset_index()
    )
    expected_datapoints_df = pd.DataFrame(expected_datapoints).reset_index()
    expected_results_df = pd.DataFrame(expected_results).reset_index()
    assert KOLENA_DEFAULT_ID in uploaded_datapoints.columns
    assert KOLENA_DEFAULT_ID in uploaded_results.columns
    assert_frame_equal(
        uploaded_datapoints[["a", "b", "e", "params"]],
        expected_datapoints_df[["a", "b", "e", "params"]],
        check_dtype=False,
        check_like=True,
    )
    assert_frame_equal(
        uploaded_results[["sum", "str"]],
        expected_results_df[["sum", "str"]],
        check_dtype=False,
        check_like=True,
    )


def test__kolena_trace_with_time_and_multiple_models() -> None:
    run_id = uuid.uuid4().hex
    dataset_name = "test__kolena_trace_with_time_and_default_id" + run_id
    model_name_1 = "test__kolena_trace_with_time_and_default_id_1" + run_id
    model_name_2 = "test__kolena_trace_with_time_and_default_id_2" + run_id
    expected_datapoints = []
    expected_results_1 = []
    expected_results_2 = []

    @kolena_trace(dataset_name=dataset_name, model_name_field="model", sync_interval=5, id_fields=["a"])
    def predict(model, a, b, e=2, params=None):
        time.sleep(1)
        return {"sum": a + b + e + random.random(), "str": str(f"received _result {a + b + e}")}

    for i in range(20):
        result_1 = predict(model_name_1, i, b=i + 1, params={"a": i, "b": i + 1, "c": i + 2})
        result_2 = predict(model_name_2, i, b=i + 1, params={"a": i, "b": i + 1, "c": i + 2})
        expected_results_1.append(result_1)
        expected_results_2.append(result_2)
        expected_datapoints.append(
            {
                "model": model_name_1,
                "a": i,
                "b": i + 1,
                "e": 2,
                "params": {"a": i, "b": i + 1, "c": i + 2},
            },
        )

    predict._clean_up()

    uploaded_datapoints = download_dataset(dataset_name).sort_values(by=["a"], ignore_index=True).reset_index()
    uploaded_results_1 = (
        download_results(dataset_name, model_name_1)[1][0]
        .results.sort_values(by=["sum"], ignore_index=True)
        .reset_index()
    )
    uploaded_results_2 = (
        download_results(dataset_name, model_name_2)[1][0]
        .results.sort_values(by=["sum"], ignore_index=True)
        .reset_index()
    )
    expected_datapoints_df = pd.DataFrame(expected_datapoints)
    expected_results_df_1 = pd.DataFrame(expected_results_1)
    expected_results_df_2 = pd.DataFrame(expected_results_2)
    assert KOLENA_TIMESTAMP_KEY in uploaded_datapoints.columns
    assert KOLENA_TIMESTAMP_KEY in uploaded_results_1.columns
    assert KOLENA_TIMESTAMP_KEY in uploaded_results_2.columns
    assert KOLENA_TIME_ELAPSED_KEY in uploaded_results_1.columns
    assert KOLENA_TIME_ELAPSED_KEY in uploaded_results_2.columns
    assert_frame_equal(
        uploaded_datapoints[["model", "a", "b", "e", "params"]],
        expected_datapoints_df[["model", "a", "b", "e", "params"]],
        check_dtype=False,
    )
    assert_frame_equal(uploaded_results_1[["sum", "str"]], expected_results_df_1[["sum", "str"]], check_dtype=False)
    assert_frame_equal(uploaded_results_2[["sum", "str"]], expected_results_df_2[["sum", "str"]], check_dtype=False)


def test__kolena_trace_multithreading() -> None:
    run_id = uuid.uuid4().hex
    dataset_name = "test__kolena_trace_multithreading" + run_id

    @kolena_trace(dataset_name=dataset_name, id_fields=["request_id"], model_name_field="model_name", sync_interval=5)
    def predict(data, request_id, model_name):
        time.sleep(random.random())
        return {"sum": sum(data), "mean": sum(data) / len(data), "data": data}

    # Worker function to call the predict function multiple times
    req_id = 0
    lock = threading.Lock()

    def get_request_id():
        with lock:
            nonlocal req_id
            cid = req_id
            req_id += 1
            return cid

    def worker(thread_id, num_calls):
        for i in range(num_calls):
            req_id = get_request_id()
            predict([i, i + 1], req_id, f"{dataset_name}_model_{thread_id}")

    threads = []
    for i in range(3):
        thread = threading.Thread(target=worker, args=(i, 20))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    predict._clean_up()

    uploaded_datapoints = download_dataset(dataset_name)
    assert len(uploaded_datapoints) == 60
    result1 = download_results(dataset_name, f"{dataset_name}_model_0")[1][0].results
    result2 = download_results(dataset_name, f"{dataset_name}_model_1")[1][0].results
    result3 = download_results(dataset_name, f"{dataset_name}_model_2")[1][0].results
    result1 = result1[result1["sum"].notna()]
    result2 = result2[result2["sum"].notna()]
    result3 = result3[result3["sum"].notna()]
    assert len(result1) == 20
    assert len(result2) == 20
    assert len(result3) == 20
    request_ids = list(result1["request_id"]) + list(result2["request_id"]) + list(result3["request_id"])
    assert sorted(list(uploaded_datapoints["request_id"])) == sorted(request_ids)
