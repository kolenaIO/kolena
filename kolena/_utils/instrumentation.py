import dataclasses
import datetime
import functools
import inspect
import traceback as tb
from abc import ABCMeta
from enum import Enum
from typing import Any
from typing import Callable

import kolena
from kolena._api.v1.client_log import ClientLog as API
from kolena._utils import krequests
from kolena._utils.state import _client_state


class DatadogLogLevels(str, Enum):
    ERROR = "error"


def telemetry(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not _client_state.telemetry:  # no extra exception handling unless enabled
            return func(*args, **kwargs)
        try:
            return func(*args, **kwargs)
        except BaseException as e:
            log_telemetry(e)
            raise e

    return wrapper


def upload_log(message: str, status: str) -> None:
    request = API.UploadLogRequest(
        client_version=kolena.__version__,
        timestamp=str(datetime.datetime.now()),
        message=message,
        status=status,
    )
    krequests.post(endpoint_path=API.Path.UPLOAD, json=dataclasses.asdict(request))


def log_telemetry(e: BaseException) -> None:
    try:
        stack = tb.format_stack()
        exc_format = tb.format_exception(None, e, e.__traceback__)
        combined = stack + exc_format
        upload_log("".join(combined), DatadogLogLevels.ERROR)
    except BaseException:
        """
        Attempting to upload the telemetry is best-effort. We don't want to have exceptions in that
        process be thrown to the customer--instead they should get their original stacktrace.
        """
        ...


class WithTelemetry(metaclass=ABCMeta):
    """
    Applies ``@telemetry`` to each "public" (non-underscored) service method declared on a subclass.
    """

    def __init_subclass__(cls) -> None:
        for key, value in cls.__dict__.items():
            if key.startswith("_") and not key == "__init__" or not callable(value) or inspect.isclass(value):
                continue
            setattr(cls, key, telemetry(value))
