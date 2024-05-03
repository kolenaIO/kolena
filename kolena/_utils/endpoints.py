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
import json
from typing import Any
from typing import Dict
from urllib.parse import urlencode
from urllib.parse import urlparse

from lzstring import LZString

from kolena._utils.state import _ClientState
from kolena._utils.state import get_client_state
from kolena._utils.state import get_endpoint_with_baseurl
from kolena.errors import InvalidClientStateError


def get_endpoint(endpoint_path: str, api_version: str) -> str:
    client_state = get_client_state()
    if client_state.base_url is None:
        raise InvalidClientStateError("missing base_url")
    return get_endpoint_with_baseurl(client_state.base_url, endpoint_path, api_version)


def get_platform_url() -> str:
    return _get_platform_url(get_client_state())


# pure function for testing
def _get_platform_url(client_state: _ClientState) -> str:
    return f"{_get_platform_origin(client_state)}/{client_state.tenant}"


def _get_platform_origin(client_state: _ClientState) -> str:
    if client_state.base_url is None:
        raise InvalidClientStateError("missing base_url")
    base_url = urlparse(client_state.base_url)
    if base_url.hostname == "localhost":
        return "http://localhost:3000"
    hostname = base_url.hostname or ""
    gateway_subdomain, *_ = hostname.split(".")
    subdomain = "trunk" if "trunk" in gateway_subdomain else "app"
    return f"https://{subdomain}.kolena.com"


def get_results_url(workflow: str, model_id: int, test_suite_id: int) -> str:
    return _get_results_url(get_client_state(), workflow, model_id, test_suite_id)


# pure function for testing
def _get_results_url(client_state: _ClientState, workflow: str, model_id: int, test_suite_id: int) -> str:
    from kolena._api.v1.workflow import WorkflowType  # deferred import to avoid circular dependencies

    platform_url = _get_platform_url(client_state)
    params: Dict[str, Any] = dict(modelIds=model_id, testSuiteId=test_suite_id)
    if workflow == WorkflowType.FR.value:
        path = "results/fr"
    elif workflow == WorkflowType.DETECTION.value:
        path = "results/object-detection"
    elif workflow == WorkflowType.CLASSIFICATION.value:
        path = "results/classification"
    else:
        path = "results"
        params["workflow"] = workflow
    return f"{platform_url}/{path}?{urlencode(params)}"


def get_test_suite_url(test_suite_id: int) -> str:
    return _get_test_suite_url(get_client_state(), test_suite_id)


def _get_test_suite_url(client_state: _ClientState, test_suite_id: int) -> str:
    platform_url = _get_platform_url(client_state)
    return f"{platform_url}/testing?{urlencode(dict(testSuiteId=test_suite_id))}"


def _get_dataset_url(client_state: _ClientState, dataset_id: int) -> str:
    platform_url = _get_platform_url(client_state)
    return f"{platform_url}/dataset?{urlencode(dict(datasetId=dataset_id))}"


def get_dataset_url(dataset_id: int) -> str:
    return _get_dataset_url(get_client_state(), dataset_id)


def get_model_url(model_id: int) -> str:
    return _get_model_url(get_client_state(), model_id)


def _get_model_url(client_state: _ClientState, model_id: int) -> str:
    platform_url = _get_platform_url(client_state)
    return f"{platform_url}/models?{urlencode(dict(modelIds=model_id))}"


def serialize_models_url(model_id: int, eval_config_id: int) -> str:
    # Compress and encode to a URL-safe base64 string using LZString
    return LZString.compressToBase64(
        json.dumps(
            {
                "id": model_id,
                "configuration": {
                    "id": eval_config_id,
                },
            },
        ),
    )
