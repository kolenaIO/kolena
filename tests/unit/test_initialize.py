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
import concurrent.futures
import dataclasses
import hashlib
import json
import time
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Optional
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from requests import Response

import kolena
from kolena._api.v1.token import ValidateResponse
from kolena._utils.consts import KOLENA_TOKEN_ENV
from kolena._utils.state import _client_state
from kolena._utils.state import _get_api_base_url
from kolena._utils.state import API_URL
from kolena._utils.state import API_URL_ENV_VAR
from kolena._utils.state import get_client_state
from kolena._utils.state import get_endpoint_with_baseurl
from kolena._utils.state import kolena_session
from kolena.dataset import download_dataset
from kolena.errors import MissingTokenError
from kolena.errors import UninitializedError


def mock_jwt(input: str) -> str:
    return hashlib.sha256(bytes(input, "utf-8")).hexdigest()


def mock_get_token(api_token: str, *args: Any, **kwargs: Any) -> ValidateResponse:
    return ValidateResponse(
        tenant="mock tenant",
        access_token=mock_jwt(api_token),
        token_type="mock token type",
        tenant_telemetry=False,
    )


def mock_get_token_with_telemetry(api_token: str, *args: Any, **kwargs: Any) -> ValidateResponse:
    return ValidateResponse(
        tenant="mock tenant",
        access_token=mock_jwt(api_token),
        token_type="mock token type",
        tenant_telemetry=True,
    )


MOCK_TOKEN = "mock token"
FIXED_TOKEN_RESPONSE = mock_get_token(MOCK_TOKEN)
FIXED_TOKEN_RESPONSE_WITH_TELEMETRY = mock_get_token_with_telemetry(MOCK_TOKEN)


@pytest.fixture(autouse=True)
def clean_client_state() -> Iterator[None]:
    try:
        yield
    finally:
        _client_state.reset()


def test__initialize__positional() -> None:
    with patch("kolena._utils.state.get_token", return_value=FIXED_TOKEN_RESPONSE):
        kolena.initialize("bar")
        assert _client_state.api_token == "bar"
        assert _client_state.jwt_token is not None
        assert not _client_state.telemetry
        assert _client_state.additional_request_headers is None


def test__initialize__telemetry() -> None:
    with patch("kolena._utils.state.get_token", return_value=FIXED_TOKEN_RESPONSE_WITH_TELEMETRY):
        kolena.initialize(api_token="bar")
        assert _client_state.api_token == "bar"
        assert _client_state.jwt_token is not None
        assert _client_state.telemetry
        assert _client_state.additional_request_headers is None


def test__initialize__keyword() -> None:
    with patch("kolena._utils.state.get_token", return_value=FIXED_TOKEN_RESPONSE):
        kolena.initialize(api_token="foo")
        assert _client_state.api_token == "foo"
        assert _client_state.jwt_token is not None
        assert _client_state.additional_request_headers is None


def test__initialize__deprecated_positional() -> None:
    with patch("kolena._utils.state.get_token", return_value=FIXED_TOKEN_RESPONSE):
        kolena.initialize("random entity", "def")
        assert _client_state.api_token == "def"
        assert _client_state.jwt_token is not None


@pytest.mark.parametrize(
    "env,expected",
    [
        ({}, API_URL),
        ({API_URL_ENV_VAR: "foobar"}, "foobar"),
        ({API_URL_ENV_VAR: ""}, API_URL),
        ({"KOLENA_MODEL": "my model"}, API_URL),
    ],
)
def test__initialize__api_url_environ(env: Dict[str, str], expected: str) -> None:
    with patch.dict("os.environ", env, clear=True):
        assert _get_api_base_url() == expected


def test__initialize__updated_environ() -> None:
    base_url = "https://internal-api.kolena.io"
    mock_response = Response()
    mock_response.status_code = 200
    mock_response._content = str.encode(json.dumps(dataclasses.asdict(FIXED_TOKEN_RESPONSE)))
    with patch.dict("os.environ", {API_URL_ENV_VAR: base_url}, clear=True):
        with patch("requests.put", return_value=mock_response) as patched:
            kolena.initialize(api_token="def")
            client_state = get_client_state()
            assert client_state.api_token == "def"

        patched.assert_called_once()
        assert patched.call_args.args[0] != get_endpoint_with_baseurl(base_url, "token/login")


def test__initialize__deprecated_keyword() -> None:
    with patch("kolena._utils.state.get_token", return_value=FIXED_TOKEN_RESPONSE):
        kolena.initialize(entity="test entity", api_token="abc")
        assert _client_state.api_token == "abc"
        assert _client_state.jwt_token is not None


@patch.dict("os.environ", {KOLENA_TOKEN_ENV: "abc"}, True)
def test__initialize__token_fallback_environ() -> None:
    with patch("kolena._utils.state.get_token", return_value=FIXED_TOKEN_RESPONSE):
        kolena.initialize()
        assert _client_state.api_token == "abc"
        assert _client_state.jwt_token is not None


@patch("netrc.netrc")
@patch.dict("os.environ", clear=True)
def test__initialize__token_fallback_netrc(mock_netrc: Mock) -> None:
    with patch("kolena._utils.state.get_token", return_value=FIXED_TOKEN_RESPONSE):
        mock_netrc.return_value.authenticators.return_value = None, None, "abc"
        kolena.initialize()
        assert _client_state.api_token == "abc"
        assert _client_state.jwt_token is not None


@patch.dict("os.environ", clear=True)
def test__initialize__token_missing() -> None:
    with patch("netrc.netrc", side_effect=MissingTokenError):
        with pytest.raises(MissingTokenError):
            kolena.initialize()


def test__uninitialized_usage() -> None:
    with pytest.raises(UninitializedError):
        download_dataset("does not exist")


def test__kolena_session() -> None:
    base_token = "foobar"
    token_1 = "token tenant one"
    token_2 = "token tenant two"
    additional_headers_1 = {"additional_header_key_one": "additional_header_val_one"}
    additional_headers_2 = {"additional_header_key_two": "additional_header_val_two"}
    proxies_2 = {"http": "dummy-proxy"}

    def check_global_client_state(
        api_token: Optional[str],
        jwt_set: bool,
        additional_request_headers: Optional[Dict[str, Any]] = None,
    ) -> None:
        assert _client_state.api_token == api_token
        assert _client_state.additional_request_headers == additional_request_headers
        if jwt_set:
            assert _client_state.jwt_token is not None
        else:
            assert _client_state.jwt_token is None

    with patch("kolena._utils.state.get_token", side_effect=mock_get_token):
        with kolena_session(token_1) as new_client_state:
            check_global_client_state(None, False)
            assert new_client_state.api_token == token_1
            assert new_client_state.jwt_token == mock_jwt(token_1)
            assert new_client_state.additional_request_headers is None
            assert new_client_state.proxies == {}
            _client_state.update(additional_request_headers=additional_headers_1)
            check_global_client_state(None, False, additional_headers_1)

            with kolena_session(
                token_2,
                additional_request_headers=additional_headers_2,
                proxies=proxies_2,
            ) as inner_state:
                check_global_client_state(None, False, additional_headers_1)
                assert inner_state.api_token == token_2
                assert inner_state.jwt_token == mock_jwt(token_2)
                assert inner_state.additional_request_headers == additional_headers_2
                assert inner_state.proxies == proxies_2

                # outer client state should stay the same
                assert new_client_state.api_token == token_1
                assert new_client_state.jwt_token == mock_jwt(token_1)
                assert new_client_state.additional_request_headers is None
                assert new_client_state.proxies == {}

            kolena.initialize(api_token=base_token)
            check_global_client_state(base_token, True, additional_headers_1)

            # context-client-state should be used in making SDK requests
            # white-boxy indirect verification
            assert get_client_state() == new_client_state

        # verify context closing does not change global client state
        check_global_client_state(base_token, True, additional_headers_1)
        assert get_client_state() == _client_state


def test__kolena_session__multithread() -> None:
    base_token = "foobar"
    token_1 = "token tenant one"
    token_2 = "token tenant two"

    def run_with_session(api_token: str, wait: float) -> None:
        with kolena_session(api_token=api_token) as client_state:
            time.sleep(wait)
            assert client_state.api_token == api_token
            assert _client_state.api_token != client_state.api_token
            assert get_client_state() == client_state

    with patch("kolena._utils.state.get_token", side_effect=mock_get_token):
        kolena.initialize(api_token=base_token)

        # check thread state is clean
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = [
                executor.submit(run_with_session, token_1, 0.5),
                executor.submit(run_with_session, token_2, 0.5),
            ]

            for future in futures:
                future.result()

        # check thread states do not interfere
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(run_with_session, token_1, 1),
                executor.submit(run_with_session, token_2, 1),
            ]

            for future in futures:
                future.result()
