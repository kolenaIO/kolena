from typing import Union
from urllib.parse import urlparse

import pytest

from kolena._api.v1.workflow import WorkflowType
from kolena._utils.endpoints import _get_platform_url
from kolena._utils.endpoints import _get_results_url
from kolena._utils.state import _ClientState
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
        (_ClientState(tenant="test-tenant"), "https://app.kolena.io/test-tenant"),
        (
            _ClientState(tenant="test-tenant", base_url="https://trunk-gateway.kolena.cloud"),
            "https://trunk.kolena.io/test-tenant",
        ),
        (
            _ClientState(tenant="test-tenant", base_url="http://localhost:8000"),
            "http://localhost:3000/test-tenant",
        ),
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
            "https://app.kolena.io/test-tenant/results/fr?modelIds=1&testSuiteIds=2",
        ),
        (
            _ClientState(tenant="test-tenant", base_url="https://trunk-gateway.kolena.cloud"),
            WorkflowType.FR.value,
            "https://trunk.kolena.io/test-tenant/results/fr?modelIds=1&testSuiteIds=2",
        ),
        (
            _ClientState(tenant="test-tenant", base_url="http://localhost:8000"),
            WorkflowType.FR.value,
            "http://localhost:3000/test-tenant/results/fr?modelIds=1&testSuiteIds=2",
        ),
        (
            _ClientState(tenant="test-tenant"),
            WorkflowType.DETECTION.value,
            "https://app.kolena.io/test-tenant/results/object-detection?modelIds=1&testSuiteIds=2",
        ),
        (
            _ClientState(tenant="test-tenant"),
            WorkflowType.CLASSIFICATION.value,
            "https://app.kolena.io/test-tenant/results/classification?modelIds=1&testSuiteIds=2",
        ),
        (
            _ClientState(tenant="test-tenant"),
            "example-generic-workflow",
            "https://app.kolena.io/test-tenant/results?modelIds=1&testSuiteIds=2&workflow=example-generic-workflow",
        ),
        (
            _ClientState(tenant="test-tenant"),
            "Example Generic Workflow",
            "https://app.kolena.io/test-tenant/results?modelIds=1&testSuiteIds=2&workflow=Example+Generic+Workflow",
        ),
    ],
)
def test__get_results_url(client_state: _ClientState, workflow: Union[Workflow, WorkflowType], expected: str) -> None:
    assert_url_equals(_get_results_url(client_state, workflow, 1, 2), expected)
