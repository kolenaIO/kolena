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
import warnings
from typing import Any
from typing import Dict
from typing import Optional

import pandas as pd

from kolena._utils import log
from kolena._utils import state
from kolena._utils.endpoints import get_platform_url
from kolena._utils.instrumentation import upload_log
from kolena._utils.state import _client_state
from kolena.errors import InputValidationError


def initialize(
    api_token: str,
    *args: Any,
    verbose: bool = False,
    proxies: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> None:
    """
    Initialize a client session.

    A session has a global scope and remains active until interpreter shutdown.

    .. note:: As of version 0.29.0: the entity param is no longer needed; ``initialize(entity, token)`` is
        **deprecated** and replaced by ``initialize(token)``.

    :param api_token: provided API token. This token is a secret and should be treated with caution
    :param verbose: optionally configure client to run in verbose mode, providing more information about execution. All
        logging events are emitted as Python standard library ``logging`` events from the ``"kolena"`` logger as well as
        to stdout/stderr directly
    :param proxies: optionally configure client to run with ``http`` or ``https`` proxies. The ``proxies`` parameter
        is passed through to the ``requests`` package and can be
        `configured accordingly <https://requests.readthedocs.io/en/latest/user/advanced/#proxies>`_
    :raises InvalidTokenError: the provided ``api_token`` is not valid
    :raises InputValidationError: the provided combination or number of args is not valid
    """
    used_deprecated_signature = False

    if len(args) > 1:
        raise InputValidationError(f"Too many args. Expected 0 or 1 but got {len(args)} Check docs for usage.")
    elif len(args) == 1:
        # overwrite the originally passed api_token since we are supporting backward compatability with entity
        api_token = args[0]
    if len(args) == 1 or "entity" in kwargs:
        used_deprecated_signature = True
        warnings.warn(
            "The signature initialize(entity, token) is deprecated. Please update to initialize(token).",
            category=DeprecationWarning,
            stacklevel=2,
        )

    init_response = state.get_token(api_token, proxies=proxies)
    derived_telemetry = init_response.tenant_telemetry
    _client_state.update(
        api_token=api_token,
        jwt_token=init_response.access_token,
        tenant=init_response.tenant,
        verbose=verbose,
        telemetry=derived_telemetry,
        proxies=proxies,
    )

    if used_deprecated_signature:
        upload_log("Client attempted to use deprecated entity auth signature.", "warn")

    log.info("initialized")
    if verbose:
        # Configure third party logging based on verbosity
        pd.set_option("display.max_colwidth", None)
        log.info(f"connected to {get_platform_url()}")
