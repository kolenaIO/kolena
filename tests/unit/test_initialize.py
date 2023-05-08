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
from typing import Iterator
from unittest.mock import patch

import pytest

import kolena
from kolena._api.v1.token import ValidateResponse
from kolena._utils.state import _client_state
from kolena.errors import UninitializedError


MOCK_TOKEN = "mock token"
FIXED_TOKEN_RESPONSE = ValidateResponse(
    tenant="mock tenant",
    access_token=MOCK_TOKEN,
    token_type="mock token type",
    tenant_telemetry=False,
)


@pytest.fixture
def clean_client_state() -> Iterator[None]:
    try:
        yield
    finally:
        _client_state.reset()


def test_initialize_positional(clean_client_state: None) -> None:
    with patch("kolena._utils.state.get_token", return_value=FIXED_TOKEN_RESPONSE):
        kolena.initialize("bar")
        assert _client_state.api_token == "bar"
        assert _client_state.jwt_token is not None


def test_initialize_keyword(clean_client_state: None) -> None:
    with patch("kolena._utils.state.get_token", return_value=FIXED_TOKEN_RESPONSE):
        kolena.initialize(api_token="foo")
        assert _client_state.api_token == "foo"
        assert _client_state.jwt_token is not None


def test_initialize_deprecated_positional(clean_client_state: None) -> None:
    with patch("kolena._utils.state.get_token", return_value=FIXED_TOKEN_RESPONSE):
        kolena.initialize("random entity", "def")
        assert _client_state.api_token == "def"
        assert _client_state.jwt_token is not None


def test_initialize_deprecated_keyword(clean_client_state: None) -> None:
    with patch("kolena._utils.state.get_token", return_value=FIXED_TOKEN_RESPONSE):
        kolena.initialize(entity="test entity", api_token="abc")
        assert _client_state.api_token == "abc"
        assert _client_state.jwt_token is not None


def test_uninitialized_usage(clean_client_state: None) -> None:
    with pytest.raises(UninitializedError):
        kolena.fr.Model.create("test", {})
