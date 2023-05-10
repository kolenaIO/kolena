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
from kolena._utils.endpoints import _get_platform_url
from kolena._utils.endpoints import _get_results_url
from kolena._utils.state import ClientState
from kolena.workflow import Workflow


def assert_url_equals(a: str, b: str) -> None:
    url_a = urlparse(a)
    url_b = urlparse(b)
    assert url_a.scheme == url_b.scheme
    assert url_a.hostname == url_b.hostname
    assert url_a.port == url_b.port
    assert url_a.path == url_b.path
    assert url_a.params == url_b.params


@pytest.mark.parametrize(
    "client_state,expected",
    [
        (ClientState(tenant="test-tenant"), "https://app.kolena.io/test-tenant"),
        (
            ClientState(tenant="test-tenant", base_url="https://trunk-api.kolena.io"),
            "https://trunk.kolena.io/test-tenant",
        ),
        (
            ClientState(tenant="test-tenant", base_url="http://localhost:8000"),
            "http://localhost:3000/test-tenant",
        ),
    ],
)
def test__get_platform_url(client_state: ClientState, expected: str) -> None:
    assert_url_equals(_get_platform_url(client_state), expected)


@pytest.mark.parametrize(
    "client_state,workflow,expected",
    [
        (
            ClientState(tenant="test-tenant"),
            WorkflowType.FR.value,
            "https://app.kolena.io/test-tenant/results/fr?modelIds=1&testSuiteIds=2",
        ),
        (
            ClientState(tenant="test-tenant", base_url="https://trunk-api.kolena.io"),
            WorkflowType.FR.value,
            "https://trunk.kolena.io/test-tenant/results/fr?modelIds=1&testSuiteIds=2",
        ),
        (
            ClientState(tenant="test-tenant", base_url="http://localhost:8000"),
            WorkflowType.FR.value,
            "http://localhost:3000/test-tenant/results/fr?modelIds=1&testSuiteIds=2",
        ),
        (
            ClientState(tenant="test-tenant"),
            WorkflowType.DETECTION.value,
            "https://app.kolena.io/test-tenant/results/object-detection?modelIds=1&testSuiteIds=2",
        ),
        (
            ClientState(tenant="test-tenant"),
            WorkflowType.CLASSIFICATION.value,
            "https://app.kolena.io/test-tenant/results/classification?modelIds=1&testSuiteIds=2",
        ),
        (
            ClientState(tenant="test-tenant"),
            "example-generic-workflow",
            "https://app.kolena.io/test-tenant/results?modelIds=1&testSuiteIds=2&workflow=example-generic-workflow",
        ),
        (
            ClientState(tenant="test-tenant"),
            "Example Generic Workflow",
            "https://app.kolena.io/test-tenant/results?modelIds=1&testSuiteIds=2&workflow=Example+Generic+Workflow",
        ),
    ],
)
def test__get_results_url(client_state: ClientState, workflow: Union[Workflow, WorkflowType], expected: str) -> None:
    assert_url_equals(_get_results_url(client_state, workflow, 1, 2), expected)
