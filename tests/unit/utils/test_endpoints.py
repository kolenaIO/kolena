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
from typing import Union
from urllib.parse import urlparse

import pytest

from kolena._api.v1.workflow import WorkflowType
from kolena._utils.endpoints import _get_model_url
from kolena._utils.endpoints import _get_platform_url
from kolena._utils.endpoints import _get_results_url
from kolena._utils.endpoints import _get_test_suite_url
from kolena._utils.state import _ClientState
from kolena.workflow import Workflow

BASE_URL_TRUNK = "https://trunk-api.kolena.io"
BASE_URL_LOCALHOST = "http://localhost:8000"


def assert_url_equals(a: str, b: str) -> None:
    url_a = urlparse(a)
    url_b = urlparse(b)
    assert url_a.scheme == url_b.scheme
    assert url_a.hostname == url_b.hostname
    assert url_a.port == url_b.port
    assert url_a.path == url_b.path
    assert url_a.params == url_b.params
    assert url_a.query == url_b.query


@pytest.mark.parametrize(
    "client_state,expected",
    [
        (_ClientState(tenant="test-tenant"), "https://app.kolena.io/test-tenant"),
        (_ClientState(tenant="test-tenant", base_url=BASE_URL_TRUNK), "https://trunk.kolena.io/test-tenant"),
        (_ClientState(tenant="test-tenant", base_url=BASE_URL_LOCALHOST), "http://localhost:3000/test-tenant"),
    ],
)
def test__get_platform_url(client_state: _ClientState, expected: str) -> None:
    assert_url_equals(_get_platform_url(client_state), expected)


@pytest.mark.parametrize(
    "client_state,workflow,expected",
    [
        (
            _ClientState(tenant="test-tenant"),
            WorkflowType.FR.value,
            "https://app.kolena.io/test-tenant/results/fr?modelIds=1&testSuiteId=2",
        ),
        (
            _ClientState(tenant="test-tenant", base_url=BASE_URL_TRUNK),
            WorkflowType.FR.value,
            "https://trunk.kolena.io/test-tenant/results/fr?modelIds=1&testSuiteId=2",
        ),
        (
            _ClientState(tenant="test-tenant", base_url=BASE_URL_LOCALHOST),
            WorkflowType.FR.value,
            "http://localhost:3000/test-tenant/results/fr?modelIds=1&testSuiteId=2",
        ),
        (
            _ClientState(tenant="test-tenant"),
            WorkflowType.DETECTION.value,
            "https://app.kolena.io/test-tenant/results/object-detection?modelIds=1&testSuiteId=2",
        ),
        (
            _ClientState(tenant="test-tenant"),
            WorkflowType.CLASSIFICATION.value,
            "https://app.kolena.io/test-tenant/results/classification?modelIds=1&testSuiteId=2",
        ),
        (
            _ClientState(tenant="test-tenant"),
            "example-workflow",
            "https://app.kolena.io/test-tenant/results?modelIds=1&testSuiteId=2&workflow=example-workflow",
        ),
        (
            _ClientState(tenant="test-tenant"),
            "Example Workflow",
            "https://app.kolena.io/test-tenant/results?modelIds=1&testSuiteId=2&workflow=Example+Workflow",
        ),
    ],
)
def test__get_results_url(client_state: _ClientState, workflow: Union[Workflow, WorkflowType], expected: str) -> None:
    assert_url_equals(_get_results_url(client_state, workflow, 1, 2), expected)


@pytest.mark.parametrize(
    "client_state,expected",
    [
        (_ClientState(tenant="a"), "https://app.kolena.io/a/testing?testSuiteId=1"),
        (_ClientState(tenant="b", base_url=BASE_URL_TRUNK), "https://trunk.kolena.io/b/testing?testSuiteId=1"),
        (_ClientState(tenant="c", base_url=BASE_URL_LOCALHOST), "http://localhost:3000/c/testing?testSuiteId=1"),
    ],
)
def test__get_test_suite_url(client_state: _ClientState, expected: str) -> None:
    assert_url_equals(_get_test_suite_url(client_state, 1), expected)


@pytest.mark.parametrize(
    "client_state,expected",
    [
        (_ClientState(tenant="a"), "https://app.kolena.io/a/models?modelIds=1"),
        (_ClientState(tenant="b", base_url=BASE_URL_TRUNK), "https://trunk.kolena.io/b/models?modelIds=1"),
        (_ClientState(tenant="c", base_url=BASE_URL_LOCALHOST), "http://localhost:3000/c/models?modelIds=1"),
    ],
)
def test__get_model_url(client_state: _ClientState, expected: str) -> None:
    assert_url_equals(_get_model_url(client_state, 1), expected)
