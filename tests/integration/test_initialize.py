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

import pytest
import requests

import kolena
from kolena._utils.state import _client_state
from kolena.errors import InvalidTokenError
from kolena.errors import RemoteError


@pytest.fixture
def clean_client_state() -> Iterator[None]:
    try:
        yield
    finally:
        _client_state.reset()


def test__initialize(clean_client_state: None, kolena_token: str) -> None:
    kolena.initialize(kolena_token)
    assert _client_state.api_token == kolena_token
    assert _client_state.jwt_token is not None


def test__initialize__deprecated_old_client(clean_client_state: None, kolena_token: str) -> None:
    """Manually test acceptance of 'entity' with raw request for client versions prior to 0.29.0"""
    url = _client_state.base_url + "/v1/token/login"
    payload = {"entity": "ignored", "version": "0.28.0", "api_token": kolena_token}
    resp = requests.put(url, json=payload)

    assert resp.status_code == 200
    assert resp.json()["access_token"] is not None


def test__initialize__invalid_version(clean_client_state: None, kolena_token: str) -> None:
    version = kolena.__version__
    try:
        kolena.__version__ = "0.0.0"
        with pytest.raises(RemoteError):
            kolena.initialize(kolena_token)
    finally:
        kolena.__version__ = version


def test__initialize__bad_token(clean_client_state: None) -> None:
    with pytest.raises(InvalidTokenError):
        kolena.initialize("bad_token")
