import uuid
from typing import Any
from typing import Dict

import requests
from requests import HTTPError
from requests_toolbelt.adapters import socket_options

from kolena import __version__ as version
from kolena._utils.endpoints import get_endpoint
from kolena._utils.state import get_client_state
from kolena._utils.state import kolena_initialized
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

STATUS_CODE__BAD_REQUEST = 400
STATUS_CODE__UNAUTHORIZED = 401
STATUS_CODE__NOT_FOUND = 404
STATUS_CODE__CONFLICT = 409

# Give the client 15 seconds to connect to kolena server
# Slightly more than a multiple of 3, as per https://docs.python-requests.org/en/master/user/advanced/#timeouts
CONNECTION_CONNECT_TIMEOUT = 15.05
CONNECTION_READ_TIMEOUT = 60 * 60  # Give kolena server 1 hour to respond to client request


@kolena_initialized
def _with_default_kwargs(**kwargs: Any) -> Dict[str, Any]:
    client_state = get_client_state()
    default_kwargs = {
        "timeout": (CONNECTION_CONNECT_TIMEOUT, CONNECTION_READ_TIMEOUT),
        "proxies": client_state.proxies,
    }
    default_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {client_state.jwt_token}",
        "X-Request-ID": uuid.uuid4().hex,
        "User-Agent": f"kolena-client/{version}",
    }
    return {
        **default_kwargs,
        **kwargs,
        "headers": {**default_headers, **kwargs.get("headers", {})},
    }


@kolena_initialized
def get(endpoint_path: str, params: Any = None, **kwargs: Any) -> requests.Response:
    url = get_endpoint(endpoint_path=endpoint_path)
    return requests.get(url=url, params=params, **_with_default_kwargs(**kwargs))


@kolena_initialized
def post(endpoint_path: str, data: Any = None, json: Any = None, **kwargs: Any) -> requests.Response:
    url = get_endpoint(endpoint_path=endpoint_path)
    with requests.Session() as s:
        s.mount("https://", socket_options.TCPKeepAliveAdapter())
        return s.post(url=url, data=data, json=json, **_with_default_kwargs(**kwargs))


@kolena_initialized
def put(endpoint_path: str, data: Any = None, json: Any = None, **kwargs: Any) -> requests.Response:
    url = get_endpoint(endpoint_path=endpoint_path)
    with requests.Session() as s:
        s.mount("https://", socket_options.TCPKeepAliveAdapter())
        return s.put(url=url, data=data, json=json, **_with_default_kwargs(**kwargs))


@kolena_initialized
def delete(endpoint_path: str, **kwargs: Any) -> requests.Response:
    url = get_endpoint(endpoint_path=endpoint_path)
    return requests.delete(url=url, **_with_default_kwargs(**kwargs))


def raise_for_status(response: requests.Response) -> None:
    if response.status_code == STATUS_CODE__UNAUTHORIZED:
        # HTTP 401 is "unauthorized" but used as "unauthenticated"
        raise UnauthenticatedError(response.content)
    if response.status_code == STATUS_CODE__NOT_FOUND:
        raise NotFoundError(response.content)
    if response.status_code == STATUS_CODE__CONFLICT:
        raise NameConflictError(response.content)

    try:
        response.raise_for_status()
    except HTTPError:
        raise RemoteError(f"{response.text} ({response.elapsed.total_seconds():0.5f} seconds elapsed)")


@kolena_initialized
def get_connection_args(**kwargs):
    client_state = get_client_state()
    return {"proxies": client_state.proxies}
