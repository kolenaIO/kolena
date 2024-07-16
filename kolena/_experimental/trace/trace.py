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
import inspect
import threading
import time
import uuid
from datetime import datetime
from typing import Callable
from typing import Optional

import pandas as pd

from kolena.dataset import upload_results
from kolena.dataset.dataset import _load_dataset_metadata
from kolena.dataset.dataset import _upload_dataset
from kolena.errors import NotFoundError

ONE_MINUTE = 10


class _Trace:
    def __init__(
        self,
        func: Callable,
        *,
        dataset_name: Optional[str] = None,
        model_name: Optional[str] = None,
        sync_interval=ONE_MINUTE,
        id_fields: Optional[list[str]] = None,
    ):
        self.func = func
        self.signature = inspect.signature(func)
        if not dataset_name:
            dataset_name = func.__name__
        self.dataset_name = dataset_name
        if not model_name:
            model_name = f"{func.__name__}_model"
        self.model_name = model_name
        try:
            self.existing_dataset = _load_dataset_metadata(self.dataset_name)
            self.id_fields = self.existing_dataset.id_fields
        except NotFoundError:
            self.existing_dataset = None
            if not id_fields:
                id_fields = ["_kolena_id"]
            self.id_fields = id_fields
        for field in self.id_fields:
            if field not in self.signature.parameters and field != "_kolena_id":
                raise ValueError(f"Id Field {field} not found in function signature")
        self.datapoints = []
        self.results = []
        self.last_update = time.time()
        self.task_ongoing = None
        self.sync_interval = sync_interval
        atexit.register(self._clean_up)

    def __call__(self, *args, **kwargs):
        arguments = self.signature.bind(*args, **kwargs).arguments
        start_time = datetime.now().isoformat()
        result = self.func(**arguments)
        end_time = datetime.now().isoformat()
        call_id = uuid.uuid4().hex
        arguments["_kolena_timestamp"] = start_time
        arguments["_kolena_id"] = call_id
        result_with_kolena_fields = {
            "_kolena_id": call_id,
            "result": result.__dict__ if hasattr(result, "__dict__") else str(result),
            "_kolena_timestamp": end_time,
        }
        self.datapoints.append(
            {key: value.__dict__ if hasattr(value, "__dict__") else str(value) for key, value in arguments.items()},
        )
        self.results.append(result_with_kolena_fields)
        if time.time() - self.last_update > self.sync_interval and (
            self.task_ongoing is None or not self.task_ongoing.is_alive()
        ):
            self.task_ongoing = threading.Thread(target=self._push_data)
            self.task_ongoing.start()
        return result

    def _push_data(self):
        try:
            dataset_df = pd.DataFrame(self.datapoints)
            result_df = pd.DataFrame(self.results)
            _upload_dataset(self.dataset_name, dataset_df, id_fields=self.id_fields, append_only=True)
            upload_results(self.dataset_name, self.model_name, result_df)
            self.last_update = time.time()
            self.datapoints = self.datapoints[dataset_df.shape[0] :]
            self.results = self.results[result_df.shape[0] :]
        except Exception as e:
            print(f"Failed to sync data: {e}")

    def _clean_up(self):
        if self.task_ongoing and self.task_ongoing.is_alive():
            self.task_ongoing.join()
        if self.datapoints:
            self._push_data()


def KolenaTrace(
    func: Optional[Callable] = None,
    *,
    dataset_name: Optional[str] = None,
    model_name: Optional[str] = None,
    sync_interval=ONE_MINUTE,
    id_fields: Optional[list[str]] = None,
):
    if func:
        return _Trace(
            func,
            dataset_name=dataset_name,
            model_name=model_name,
            sync_interval=sync_interval,
            id_fields=id_fields,
        )
    else:

        def wrapper(func):
            return _Trace(
                func,
                dataset_name=dataset_name,
                model_name=model_name,
                sync_interval=sync_interval,
                id_fields=id_fields,
            )

        return wrapper
