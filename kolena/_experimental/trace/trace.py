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
import atexit
import dataclasses
import functools
import inspect
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import numpy as np
import pandas as pd

import kolena
from kolena._utils.datatypes import DataObject
from kolena.dataset import upload_results
from kolena.dataset.dataset import _load_dataset_metadata
from kolena.dataset.dataset import _upload_dataset
from kolena.errors import NotFoundError

THIRTY_SECONDS = 30
KOLENA_DEFAULT_ID = "_kolena_id"
KOLENA_TIMESTAMP_KEY = "_kolena_timestamp"
KOLENA_TIME_ELAPSED_KEY = "_kolena_time_elapsed"


def _serialize_item(value: Any):
    if isinstance(value, (bool, str, int, float)) or value is None:
        return value
    elif isinstance(value, np.generic):  # numpy scalars are common enough to be worth specific handling
        for base_type, numpy_type in [(bool, np.bool_), (int, np.integer), (float, np.inexact)]:
            if isinstance(value, numpy_type):  # cast if there is a match, otherwise fallthrough
                return base_type(value)
    elif isinstance(value, Dict):
        return {key: _serialize_item(subvalue) for key, subvalue in value.items()}
    elif isinstance(value, DataObject):
        return value._to_dict()
    elif dataclasses.is_dataclass(value):
        return _serialize_item(dataclasses.asdict(value))
    elif hasattr(value, "__dict__"):
        return _serialize_item(value.__dict__)
    elif isinstance(value, (List, Set, Tuple, pd.Series, np.ndarray)):
        return [_serialize_item(item) for item in value]
    else:
        return str(value)


class _Trace:
    def __init__(
        self,
        func: Callable,
        *,
        dataset_name: Optional[str] = None,
        model_name: Optional[str] = None,
        model_name_field: Optional[str] = None,
        sync_interval=THIRTY_SECONDS,
        id_fields: Optional[List[str]] = None,
        record_timestamp: bool = True,
    ):
        kolena.initialize(verbose=False)
        self.func = func
        self.signature = inspect.signature(func)
        if not dataset_name:
            dataset_name = func.__name__
        self.dataset_name = dataset_name
        if not model_name:
            model_name = f"{self.dataset_name}_model"
        self.model_name = model_name
        if model_name_field:
            if model_name_field not in self.signature.parameters:
                raise ValueError(f"Model Name Field {model_name_field} not found in function signature")
        self.model_name_field = model_name_field
        try:
            self.existing_dataset = _load_dataset_metadata(self.dataset_name)
            if not self.existing_dataset:
                raise NotFoundError("dataset metadata not found")
            if id_fields and sorted(id_fields) != sorted(self.existing_dataset.id_fields):
                raise ValueError(f"Id Fields {id_fields} do not match existing dataset id fields")
            self.id_fields = self.existing_dataset.id_fields
        except NotFoundError:
            self.existing_dataset = None
            if not id_fields:
                id_fields = [KOLENA_DEFAULT_ID]
            self.id_fields = id_fields
        for field in self.id_fields:
            if field not in self.signature.parameters and field != KOLENA_DEFAULT_ID:
                raise ValueError(f"Id Field {field} not found in function signature")
        self.datapoints = []
        self.results = defaultdict(list)
        self.last_update = time.time()
        self.task_ongoing = None
        self.sync_interval = sync_interval
        self.record_timestamp = record_timestamp
        functools.update_wrapper(self, func)
        atexit.register(self._clean_up)
        self.lock = threading.Lock()

    def _add_id_fields(self, datapoint: Dict, result: Dict) -> None:
        for field in self.id_fields:
            if field == KOLENA_DEFAULT_ID:
                call_id = uuid.uuid4().hex
                datapoint[KOLENA_DEFAULT_ID] = call_id
                result[KOLENA_DEFAULT_ID] = call_id
            else:
                field_value = datapoint.get(field)
                if field_value is None:
                    raise ValueError(f"Id Field {field} cannot be None in datapoint input")
                result[field] = field_value

    def __call__(self, *args, **kwargs):
        bounded_arguments = self.signature.bind(*args, **kwargs)
        bounded_arguments.apply_defaults()
        arguments = bounded_arguments.arguments

        start_time = datetime.now()
        output = self.func(**arguments)
        end_time = datetime.now()
        datapoint = _serialize_item(arguments)
        result = _serialize_item(output)
        if not isinstance(result, dict):
            result = {"result": result}
        if self.record_timestamp:
            datapoint[KOLENA_TIMESTAMP_KEY] = start_time.isoformat()
            result[KOLENA_TIMESTAMP_KEY] = end_time.isoformat()
            result[KOLENA_TIME_ELAPSED_KEY] = (end_time - start_time).total_seconds()
        self._add_id_fields(datapoint, result)
        model_name = self.model_name
        if self.model_name_field and arguments.get(self.model_name_field) is not None:
            model_name = arguments.get(self.model_name_field)
        with self.lock:
            self.datapoints.append(datapoint)
            self.results[model_name].append(result)
        if time.time() - self.last_update > self.sync_interval and (
            self.task_ongoing is None or not self.task_ongoing.is_alive()
        ):
            self.task_ongoing = threading.Thread(target=self._push_data)
            self.task_ongoing.start()
        return output

    def _push_data(self):
        try:
            with self.lock:
                datapoint_count = len(self.datapoints)
                results_counts = {model_name: len(results) for model_name, results in self.results.items()}
            dataset_df = pd.DataFrame(self.datapoints[:datapoint_count])
            unique_dataset_df = dataset_df.drop_duplicates(subset=self.id_fields)
            _upload_dataset(self.dataset_name, unique_dataset_df, id_fields=self.id_fields, append_only=True)
            with self.lock:
                self.datapoints = self.datapoints[dataset_df.shape[0] :]
            for model_name, results in self.results.items():
                result_df = pd.DataFrame(results[: results_counts[model_name]])
                upload_results(self.dataset_name, model_name, result_df.drop_duplicates(subset=self.id_fields))
                with self.lock:
                    self.results[model_name] = self.results[model_name][result_df.shape[0] :]
            self.last_update = time.time()
        except Exception as e:
            print(f"Failed to sync data: {e}")

    def _clean_up(self):
        if self.task_ongoing and self.task_ongoing.is_alive():
            self.task_ongoing.join()
        if self.datapoints or any(self.results.values()):
            self._push_data()


def kolena_trace(
    func: Optional[Callable] = None,
    *,
    dataset_name: Optional[str] = None,
    model_name: Optional[str] = None,
    model_name_field: Optional[str] = None,
    sync_interval: int = THIRTY_SECONDS,
    id_fields: Optional[List[str]] = None,
    record_timestamp: bool = True,
):
    """
    Use this decorator to trace the function with Kolena, the input and output of this function will be
    sent as datapoints and results respectively
    For example:
    ```python3
    @kolena_trace(dataset_name="test_trace", id_fields=["request_id"], record_timestamp=False)
    def predict(data, request_id):
        pass
    ```
    OR

    ```python3
    @kolena_trace
    def predict(data, request_id):
        pass
    ```
    :param func: The function to be traced, this is auto populated when used as a decorator
    :param dataset_name: The name of the dataset to be created, if not provided the function name will be used
    :param model_name: The name of the model to be created, if not provided the function name suffixed with _model
    will be used
    :param model_name_field: The field in the input that should be used as model name,
    if this would override the model name
    :param sync_interval: The interval at which the data should be synced to the server, default is 30 seconds
    :param id_fields: The fields in the input that should be used as id fields,
    if not provided a default id field will be used
    :param record_timestamp: If True, the timestamp of the input, output, and time elapsed will be recorded,
    default is True

    """
    if func:
        return _Trace(
            func,
            dataset_name=dataset_name,
            model_name=model_name,
            model_name_field=model_name_field,
            sync_interval=sync_interval,
            id_fields=id_fields,
            record_timestamp=record_timestamp,
        )
    else:

        def wrapper(func):
            return _Trace(
                func,
                dataset_name=dataset_name,
                model_name=model_name,
                model_name_field=model_name_field,
                sync_interval=sync_interval,
                id_fields=id_fields,
                record_timestamp=record_timestamp,
            )

        return wrapper
