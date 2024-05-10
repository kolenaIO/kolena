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
import datetime
import functools
import json
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import Union

import kolena
from kolena._api.v1.client_log import ClientLog as API
from kolena._api.v1.core import TestRun as CoreAPI
from kolena._api.v1.event import EventAPI
from kolena._utils import krequests


class DatadogLogLevels(str, Enum):
    ERROR = "error"


def upload_log(message: str, status: str) -> None:
    request = API.UploadLogRequest(
        client_version=kolena.__version__,
        timestamp=str(datetime.datetime.now()),
        message=message,
        status=status,
    )
    krequests.post(endpoint_path=API.Path.UPLOAD.value, json=dataclasses.asdict(request))


def report_crash(id: int, endpoint_path: str) -> None:
    request = CoreAPI.MarkCrashedRequest(test_run_id=id)
    # note no krequests.raise_for_status -- already in crashed state
    krequests.post(endpoint_path=endpoint_path, data=json.dumps(dataclasses.asdict(request)))


def set_profile() -> None:
    try:
        krequests.put(endpoint_path=EventAPI.Path.PROFILE)
    except Exception:
        """
        Attempting to set up event profile is best-effort. We don't want to have exceptions in that
        process be thrown to the customer--instead they should get their original stacktrace.
        """
        ...


def record_event(request: EventAPI.RecordEventRequest) -> None:
    try:
        krequests.post(endpoint_path=EventAPI.Path.EVENT, json=dataclasses.asdict(request))
    except Exception:
        """
        Attempting to record event is best-effort. We don't want to have exceptions in that
        process be thrown to the customer--instead they should get their original stacktrace.
        """
        ...


def with_event(event_name: str) -> Callable:
    """function decorator to track start and end of an event"""

    def event_decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # track duration of the call, and if failed record the exception class name
            start_time = datetime.datetime.now()
            event_metadata: Dict[str, Union[str, int, float, bool, None]] = {}
            try:
                response = func(*args, **kwargs)
                return response
            except Exception as e:
                event_metadata["response_error"] = e.__class__.__name__
                raise e
            finally:
                event_metadata["duration"] = round((datetime.datetime.now() - start_time).total_seconds(), 3)
                record_event(
                    EventAPI.RecordEventRequest(event_name=event_name, additional_metadata=event_metadata),
                )

        return wrapper

    return event_decorator
