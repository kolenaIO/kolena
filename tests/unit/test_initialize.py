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
import concurrent.futures
import dataclasses
import hashlib
import json
import time
from typing import Any
from typing import Iterator
from typing import Optional
from unittest.mock import patch

import pytest
from requests import Response

import kolena
from kolena._api.v1.token import ValidateResponse
from kolena._utils.state import _client_state
from kolena._utils.state import API_BASE_URL
from kolena._utils.state import API_BASE_URL_ENV_VAR
from kolena._utils.state import get_client_state
from kolena._utils.state import get_endpoint_with_baseurl
from kolena._utils.state import kolena_session
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


MOCK_TOKEN = "mock token"
FIXED_TOKEN_RESPONSE = mock_get_token(MOCK_TOKEN)


@pytest.fixture
def clean_client_state() -> Iterator[None]:
    try:
        yield
    finally:
        _client_state.reset()


def test__initialize__positional(clean_client_state: None) -> None:
    with patch("kolena._utils.state.get_token", return_value=FIXED_TOKEN_RESPONSE):
        kolena.initialize("bar")
        assert _client_state.api_token == "bar"
        assert _client_state.jwt_token is not None


def test__initialize__keyword(clean_client_state: None) -> None:
    with patch("kolena._utils.state.get_token", return_value=FIXED_TOKEN_RESPONSE):
        kolena.initialize(api_token="foo")
        assert _client_state.api_token == "foo"
        assert _client_state.jwt_token is not None


def test__initialize__deprecated_positional(clean_client_state: None) -> None:
    with patch("kolena._utils.state.get_token", return_value=FIXED_TOKEN_RESPONSE):
        kolena.initialize("random entity", "def")
        assert _client_state.api_token == "def"
        assert _client_state.jwt_token is not None


def test__initialize__updated_environ(clean_client_state: None) -> None:
    base_url = "https://internal-api.kolena.io"
    mock_response = Response()
    mock_response.status_code = 200
    mock_response._content = str.encode(json.dumps(dataclasses.asdict(FIXED_TOKEN_RESPONSE)))
    with patch.dict("os.environ", {API_BASE_URL_ENV_VAR: base_url}, clear=True):
        with patch("requests.put", return_value=mock_response) as patched:
            kolena.initialize("random entity", "def")
            client_state = get_client_state()
            assert client_state.api_token == "def"

        patched.assert_called_once()
        assert patched.call_args.args[0] == get_endpoint_with_baseurl(API_BASE_URL, "token/login")


def test__initialize__deprecated_keyword(clean_client_state: None) -> None:
    with patch("kolena._utils.state.get_token", return_value=FIXED_TOKEN_RESPONSE):
        kolena.initialize(entity="test entity", api_token="abc")
        assert _client_state.api_token == "abc"
        assert _client_state.jwt_token is not None


def test__uninitialized_usage(clean_client_state: None) -> None:
    with pytest.raises(UninitializedError):
        kolena.fr.Model.create("test", {})


def test__kolena_session(clean_client_state: None) -> None:
    base_token = "foobar"
    token_1 = "token tenant one"
    token_2 = "token tenant two"

    def check_global_client_state(api_token: Optional[str], jwt_set: bool) -> None:
        assert _client_state.api_token == api_token
        if jwt_set:
            assert _client_state.jwt_token is not None
        else:
            assert _client_state.jwt_token is None

    with patch("kolena._utils.state.get_token", side_effect=mock_get_token):
        with kolena_session(token_1) as new_client_state:
            check_global_client_state(None, False)
            assert new_client_state.api_token == token_1
            assert new_client_state.jwt_token == mock_jwt(token_1)

            with kolena_session(token_2) as inner_state:
                check_global_client_state(None, False)
                assert inner_state.api_token == token_2
                assert inner_state.jwt_token == mock_jwt(token_2)
                assert new_client_state.api_token == token_1
                assert new_client_state.jwt_token == mock_jwt(token_1)

            kolena.initialize(base_token)
            check_global_client_state(base_token, True)

            # context-client-state should be used in making SDK requests
            # white-boxy indirect verification
            assert get_client_state() == new_client_state

        # verify context closing does not change global client state
        check_global_client_state(base_token, True)
        assert get_client_state() == _client_state


def test__kolena_session__multithread(clean_client_state: None) -> None:
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
        kolena.initialize(base_token)

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
