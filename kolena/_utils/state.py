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
from kolena._utils.serde import from_dict
from kolena.errors import InvalidTokenError
from kolena.errors import UnauthenticatedError

API_V1 = "v1"
API_V2 = "v2"
DEFAULT_API_VERSION = API_V1
API_URL = "https://api.kolena.io"
API_URL_ENV_VAR = "KOLENA_API_URL"
CLIENT_STATE: contextvars.ContextVar["_ClientState"] = contextvars.ContextVar("client_state")


class NoOpAuth(requests.auth.AuthBase):
    """Provide no-op Auth to disable requests using .netrc"""

    def __call__(self, r: Any) -> Any:
        return r


class _ClientState:
    def __init__(
        self,
        base_url: Optional[str] = API_URL,
        api_token: Optional[str] = None,
        jwt_token: Optional[str] = None,
        tenant: Optional[str] = None,
        verbose: bool = False,
        proxies: Optional[Dict[str, str]] = None,
        additional_request_headers: Optional[Dict[str, Any]] = None,
    ):
        self.base_url: Optional[str] = None
        self.api_token: Optional[str] = None
        self.jwt_token: Optional[str] = None
        self.tenant: Optional[str] = None
        self.verbose: bool = False
        self.proxies: Dict[str, str] = {}
        self.additional_request_headers: Optional[Dict[str, Any]] = None
        self.update(
            base_url=base_url,
            api_token=api_token,
            jwt_token=jwt_token,
            tenant=tenant,
            verbose=verbose,
            proxies=proxies,
            additional_request_headers=additional_request_headers,
        )

    def update(
        self,
        base_url: Optional[str] = None,
        api_token: Optional[str] = None,
        jwt_token: Optional[str] = None,
        tenant: Optional[str] = None,
        verbose: bool = False,
        proxies: Optional[Dict[str, str]] = None,
        additional_request_headers: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.base_url = base_url or self.base_url
        self.api_token = api_token or self.api_token
        self.jwt_token = jwt_token or self.jwt_token
        self.tenant = tenant or self.tenant
        self.verbose = verbose
        self.proxies = proxies or self.proxies
        self.additional_request_headers = additional_request_headers or self.additional_request_headers

    def reset(self) -> None:
        # note that base_url remains set
        self.api_token = None
        self.jwt_token = None
        self.tenant = None
        self.verbose = False
        self.additional_request_headers = None
        self.proxies = {}


def _get_api_base_url() -> str:
    return os.environ.get(API_URL_ENV_VAR) or API_URL


_client_state = _ClientState(base_url=_get_api_base_url())


def get_client_state() -> _ClientState:
    return CLIENT_STATE.get(_client_state)


def is_client_uninitialized() -> bool:
    return get_client_state().jwt_token is None


def kolena_initialized(func: Callable) -> Callable:
    """Attempts to initialize client if not initialized"""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if is_client_uninitialized():
            kolena.initialize()
        return func(*args, **kwargs)

    return wrapper


def get_endpoint_with_baseurl(base_url: str, endpoint_path: str, api_version: str = DEFAULT_API_VERSION) -> str:
    return f"{base_url}/{api_version}/{endpoint_path.lstrip('/')}"


def get_token(
    api_token: str,
    base_url: Optional[str] = None,
    proxies: Optional[Dict[str, str]] = None,
) -> API.ValidateResponse:
    base_url = base_url or get_client_state().base_url or ""
    request = API.ValidateRequest(api_token=api_token, version=kolena.__version__)
    r = requests.put(
        get_endpoint_with_baseurl(base_url, "token/login"),
        auth=NoOpAuth(),
        json=dataclasses.asdict(request),
        proxies=proxies,
    )
    try:
        kolena._utils.krequests.raise_for_status(r)
    except UnauthenticatedError as e:
        raise InvalidTokenError(e)

    init_response = from_dict(data_class=API.ValidateResponse, data=r.json())
    return init_response


@contextlib.contextmanager
def kolena_session(
    api_token: str,
    base_url: Optional[str] = None,
    additional_request_headers: Optional[Dict[str, Any]] = None,
    proxies: Optional[Dict[str, str]] = None,
) -> Iterator[_ClientState]:
    base_url = base_url or _get_api_base_url()
    init_response = get_token(api_token, base_url)
    client_state = _ClientState(
        base_url=base_url,
        api_token=api_token,
        jwt_token=init_response.access_token,
        tenant=init_response.tenant,
        additional_request_headers=additional_request_headers,
        proxies=proxies,
    )
    token = CLIENT_STATE.set(client_state)

    try:
        yield client_state
    finally:
        CLIENT_STATE.reset(token)
