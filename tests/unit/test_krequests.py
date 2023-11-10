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
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Optional
from unittest.mock import patch

import pytest

import kolena
from kolena._utils import krequests
from kolena._utils.state import _client_state
from tests.unit.test_initialize import FIXED_TOKEN_RESPONSE


DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "X-Request-ID": None,
    "User-Agent": None,
    "X-Kolena-Telemetry": "False",
}
DEFAULT_KWARGS = {"auth": None, "timeout": (15.05, 3600), "proxies": {}}


@pytest.fixture(autouse=True)
def clean_client_state() -> Iterator[None]:
    try:
        yield
    finally:
        _client_state.reset()


@pytest.mark.parametrize(
    "kwargs, proxies, additional_request_headers",
    [
        ({}, None, None),
        ({}, {}, {}),
        ({"url": "some-url", "data": {"key": "value"}}, None, None),
        ({"timeout": (1, 60)}, None, None),  # attempt to override default args
        ({}, {"http": "dummy-proxy"}, {}),
        ({}, {}, {"some-header-key": "some-header-val"}),
        ({}, {}, {"X-Kolena-Telemetry": "off"}),  # attempt to override default headers with client state
        ({"headers": {"X-Kolena-Telemetry": "off"}}, {}, {}),  # attempt to override default headers with kwargs
        ({"headers": {"Content-Type": "application/octet-stream"}}, {}, {}),  # should allow overriding Content-Type
        ({"url": "some-url", "data": {"key": "value"}}, {"http": "dummy-proxy"}, {"X-Kolena-Telemetry": "off"}),
    ],
)
def test__with_default_kwargs(
    kwargs: Dict[str, Any],
    proxies: Optional[Dict[str, str]],
    additional_request_headers: Optional[Dict[str, Any]],
) -> None:
    def _assert_dict_key_val(dictionary: Dict[str, Any], expected_key: str, expected_val: Optional[Any]) -> None:
        assert expected_key in dictionary
        if expected_val is not None:
            assert dictionary[expected_key] == expected_val

    with patch("kolena._utils.state.get_token", return_value=FIXED_TOKEN_RESPONSE):
        kolena.initialize("bar")
        # default kwargs should take priority and cannot be overriden by kwargs
        expected_kwargs = {**{key: kwargs[key] for key in kwargs if key != "headers"}, **DEFAULT_KWARGS}
        if proxies is not None:
            _client_state.update(proxies=proxies)
            expected_kwargs["proxies"] = proxies

        expected_headers = DEFAULT_HEADERS
        if additional_request_headers is not None:
            _client_state.update(additional_request_headers=additional_request_headers)
            # default headers should take priority and cannot be overriden by additional headers
            expected_headers = {**additional_request_headers, **DEFAULT_HEADERS}

        # values passed by kwargs should not override default args
        if "headers" in kwargs:
            for key in kwargs.get("headers"):
                if key not in expected_headers or key == "Content-Type":
                    expected_headers[key] = kwargs.get("headers")[key]

        default_kwargs = krequests._with_default_kwargs(**kwargs)
        for key in expected_kwargs:
            _assert_dict_key_val(default_kwargs, key, expected_kwargs[key])
        for key in expected_headers:
            _assert_dict_key_val(default_kwargs.get("headers"), key, expected_headers[key])
