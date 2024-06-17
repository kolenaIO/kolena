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
import uuid
from http import HTTPStatus
from typing import Any
from typing import Dict

import requests
from requests import HTTPError
from requests_toolbelt import user_agent
from requests_toolbelt.adapters import socket_options
from urllib3.util import Retry

from kolena import __name__ as client_name
from kolena import __version__ as client_version
from kolena._utils.endpoints import get_endpoint
from kolena._utils.state import DEFAULT_API_VERSION
from kolena._utils.state import get_client_state
from kolena._utils.state import kolena_initialized
from kolena.errors import IncorrectUsageError
from kolena.errors import NameConflictError
from kolena.errors import NotFoundError
from kolena.errors import RemoteError
from kolena.errors import UnauthenticatedError

__all__ = [
    "get",
    "post",
    "put",
    "delete",
    "raise_for_status",
]

# Give the client 15 seconds to connect to kolena server
# Slightly more than a multiple of 3, as per https://docs.python-requests.org/en/master/user/advanced/#timeouts
CONNECTION_CONNECT_TIMEOUT = 15.05
CONNECTION_READ_TIMEOUT = 60 * 60  # Give kolena server 1 hour to respond to client request

# This only retries for failed DNS lookups, socket connections and connection timeouts.
# HTTPAdapter sets this to 0 by default. https://requests.readthedocs.io/en/latest/_modules/requests/adapters/
# Using the Retry object to configure a backoff which is not supported by using an int here.
MAX_RETRIES = Retry(total=3, connect=3, read=0, redirect=0, status=0, backoff_factor=2)

# This retries for read errors in addition to the ones mentioned above.
# Should only be used for idempotent operations.
MAX_IDEMPOTENT_RETRIES = Retry(total=3, connect=3, read=3, redirect=0, status=0, backoff_factor=2)


class JWTAuth(requests.auth.AuthBase):
    """Attaches JWT Authorization to the given Request object"""

    def __init__(self, jwt: str):
        self.jwt = jwt

    def __call__(self, r: Any) -> Any:
        r.headers["Authorization"] = f"Bearer {self.jwt}"
        return r


@kolena_initialized
def _with_default_kwargs(**kwargs: Any) -> Dict[str, Any]:
    client_state = get_client_state()
    assert client_state.jwt_token is not None
    default_kwargs = {
        "auth": JWTAuth(client_state.jwt_token),
        "timeout": (CONNECTION_CONNECT_TIMEOUT, CONNECTION_READ_TIMEOUT),
        "proxies": client_state.proxies,
    }
    default_headers = client_state.additional_request_headers or {}
    default_headers.update(
        {
            "Content-Type": "application/json",
            "X-Request-ID": uuid.uuid4().hex,
            "User-Agent": user_agent(client_name, client_version),
        },
    )
    # allow requests to override Content-Type as needed by for example file uploads
    kw_headers = kwargs.get("headers", {})
    if "Content-Type" in kw_headers:
        default_headers["Content-Type"] = kw_headers["Content-Type"]
    return {
        **kwargs,
        **default_kwargs,
        "headers": {**kwargs.get("headers", {}), **default_headers},
    }


@kolena_initialized
def get(
    endpoint_path: str,
    params: Any = None,
    api_version: str = DEFAULT_API_VERSION,
    **kwargs: Any,
) -> requests.Response:
    url = get_endpoint(endpoint_path=endpoint_path, api_version=api_version)
    with requests.Session() as s:
        s.mount("https://", socket_options.TCPKeepAliveAdapter(max_retries=MAX_IDEMPOTENT_RETRIES))
        return s.get(url=url, params=params, **_with_default_kwargs(**kwargs))


@kolena_initialized
def post(
    endpoint_path: str,
    data: Any = None,
    json: Any = None,
    api_version: str = DEFAULT_API_VERSION,
    **kwargs: Any,
) -> requests.Response:
    url = get_endpoint(endpoint_path=endpoint_path, api_version=api_version)
    with requests.Session() as s:
        s.mount("https://", socket_options.TCPKeepAliveAdapter(max_retries=MAX_RETRIES))
        return s.post(url=url, data=data, json=json, **_with_default_kwargs(**kwargs))


@kolena_initialized
def put(
    endpoint_path: str,
    data: Any = None,
    json: Any = None,
    api_version: str = DEFAULT_API_VERSION,
    **kwargs: Any,
) -> requests.Response:
    url = get_endpoint(endpoint_path=endpoint_path, api_version=api_version)
    with requests.Session() as s:
        s.mount("https://", socket_options.TCPKeepAliveAdapter(max_retries=MAX_IDEMPOTENT_RETRIES))
        return s.put(url=url, data=data, json=json, **_with_default_kwargs(**kwargs))


@kolena_initialized
def delete(endpoint_path: str, api_version: str = DEFAULT_API_VERSION, **kwargs: Any) -> requests.Response:
    url = get_endpoint(endpoint_path=endpoint_path, api_version=api_version)
    with requests.Session() as s:
        s.mount("https://", socket_options.TCPKeepAliveAdapter(max_retries=MAX_IDEMPOTENT_RETRIES))
        return requests.delete(url=url, **_with_default_kwargs(**kwargs))


def raise_for_status(response: requests.Response) -> None:
    if response.status_code == HTTPStatus.UNAUTHORIZED:
        # HTTP 401 is "unauthorized" but used as "unauthenticated"
        raise UnauthenticatedError(response.content)
    if response.status_code == HTTPStatus.NOT_FOUND:
        raise NotFoundError(response.content)
    if response.status_code == HTTPStatus.CONFLICT:
        raise NameConflictError(response.content)
    if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
        raise IncorrectUsageError(response.content)

    try:
        response.raise_for_status()
    except HTTPError:
        raise RemoteError(f"{response.text} ({response.elapsed.total_seconds():0.5f} seconds elapsed)")


@kolena_initialized
def get_connection_args(**kwargs: Any) -> Dict[str, Dict[str, str]]:
    client_state = get_client_state()
    return {"proxies": client_state.proxies}
