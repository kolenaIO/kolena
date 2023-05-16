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
import contextlib
import contextvars
import dataclasses
import functools
import os
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import Optional

import requests

import kolena
import kolena._api.v1.token as API
from kolena._utils import krequests
from kolena._utils.serde import from_dict
from kolena.errors import InvalidClientStateError
from kolena.errors import InvalidTokenError
from kolena.errors import UnauthenticatedError
from kolena.errors import UninitializedError

API_VERSION = "v1"
API_BASE_URL = "https://api.kolena.io"
API_BASE_URL_ENV_VAR = "KOLENA_CLIENT_BASE_URL"
CLIENT_STATE = contextvars.ContextVar("client_state")


class _ClientState:
    def __init__(
        self,
        base_url: Optional[str] = API_BASE_URL,
        api_token: Optional[str] = None,
        jwt_token: Optional[str] = None,
        tenant: Optional[str] = None,
        verbose: bool = False,
        telemetry: bool = False,
        proxies: Optional[Dict[str, str]] = None,
    ):
        self.base_url: Optional[str] = None
        self.api_token: Optional[str] = None
        self.jwt_token: Optional[str] = None
        self.tenant: Optional[str] = None
        self.verbose: bool = False
        self.telemetry: bool = False
        self.proxies: Dict[str, str] = {}
        self.update(
            base_url=base_url,
            api_token=api_token,
            jwt_token=jwt_token,
            tenant=tenant,
            verbose=verbose,
            telemetry=telemetry,
            proxies=proxies,
        )

    def update(
        self,
        base_url: Optional[str] = None,
        api_token: Optional[str] = None,
        jwt_token: Optional[str] = None,
        tenant: Optional[str] = None,
        verbose: bool = False,
        telemetry: bool = False,
        proxies: Optional[Dict[str, str]] = None,
    ) -> None:
        self.base_url = base_url or self.base_url
        self.api_token = api_token or self.api_token
        self.jwt_token = jwt_token or self.jwt_token
        self.tenant = tenant or self.tenant
        self.verbose = verbose
        self.telemetry = telemetry
        self.proxies = proxies or {}

    def assert_initialized(self) -> None:
        if self.base_url is None:
            raise InvalidClientStateError("missing base_url")
        if self.jwt_token is None:
            raise UninitializedError("client has not been initialized via kolena.initialize(...)")
        if self.api_token is None:
            raise InvalidClientStateError("missing client api_token")

    def reset(self) -> None:
        # note that base_url remains set
        self.api_token = None
        self.jwt_token = None
        self.tenant = None
        self.verbose = False
        self.telemetry = False


def _get_api_base_url() -> str:
    return os.environ.get(API_BASE_URL_ENV_VAR, API_BASE_URL)


_client_state = _ClientState(base_url=_get_api_base_url())


def get_client_state() -> _ClientState:
    return CLIENT_STATE.get(_client_state)


def is_client_initialized() -> bool:
    try:
        get_client_state().assert_initialized()
    except (InvalidClientStateError, UninitializedError):
        return False
    return True


def kolena_initialized(func: Callable) -> Callable:
    """Raises InvalidKolenaStateError if not initialized"""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        client_state = get_client_state()
        client_state.assert_initialized()
        return func(*args, **kwargs)

    return wrapper


def get_endpoint_with_baseurl(base_url: str, endpoint_path: str) -> str:
    return f"{base_url}/{API_VERSION}/{endpoint_path.lstrip('/')}"


def get_token(
    api_token: str,
    base_url: Optional[str] = None,
    proxies: Optional[Dict[str, str]] = None,
) -> API.ValidateResponse:
    base_url = base_url or get_client_state().base_url
    request = API.ValidateRequest(api_token=api_token, version=kolena.__version__)
    r = requests.put(
        get_endpoint_with_baseurl(base_url, "token/login"),
        json=dataclasses.asdict(request),
        proxies=proxies,
    )
    try:
        krequests.raise_for_status(r)
    except UnauthenticatedError as e:
        raise InvalidTokenError(e)

    init_response = from_dict(data_class=API.ValidateResponse, data=r.json())
    return init_response


@contextlib.contextmanager
def kolena_session(api_token: str, base_url: Optional[str] = None) -> Iterator[_ClientState]:
    base_url = base_url or _get_api_base_url()
    init_response = get_token(api_token, base_url)
    client_state = _ClientState(
        base_url=base_url,
        api_token=api_token,
        jwt_token=init_response.access_token,
        tenant=init_response.tenant,
    )
    token = CLIENT_STATE.set(client_state)

    try:
        yield client_state
    finally:
        CLIENT_STATE.reset(token)
