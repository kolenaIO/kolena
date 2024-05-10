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
import netrc
import os
import warnings
from typing import Any
from typing import Dict
from typing import Optional
from urllib.parse import urlparse

import pandas as pd

from kolena._api.v1.event import EventAPI
from kolena._utils import log
from kolena._utils import state
from kolena._utils.consts import KOLENA_TOKEN_ENV
from kolena._utils.endpoints import get_platform_url
from kolena._utils.instrumentation import record_event
from kolena._utils.instrumentation import set_profile
from kolena._utils.instrumentation import upload_log
from kolena._utils.state import _client_state
from kolena.errors import InputValidationError
from kolena.errors import MissingTokenError


def initialize(
    *args: Any,
    api_token: Optional[str] = None,
    verbose: bool = True,
    proxies: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> None:
    """
    Initialize a client session.

    !!! tip
        It is often not necessary to call `kolena.initialize()` directly. When a function requiring initialization is
        called, this method will be called to initialize a session with any token found in your environment.

    A session has a global scope and remains active until interpreter shutdown.

    Retrieve an API token from the [:kolena-developer-16: Developer](https://app.kolena.com/redirect/developer) page and
    make it available through one of the following options before initializing:

    1. Directly through the `api_token` keyword argument
    ```python
    import kolena

    kolena.initialize(api_token=your_token)
    ```

    2. Populate the `KOLENA_TOKEN` environment variable and call `kolena.initialize()`:
    ```bash
    export KOLENA_TOKEN="********"
    ```

    3. Store in your [`.netrc` file](https://www.gnu.org/software/inetutils/manual/html_node/The-_002enetrc-file.html)
      before calling `kolena.initialize()`:
    ```cfg title="~/.netrc"
    machine api.kolena.io password ********
    ```

    ``` mermaid
    flowchart TD
        Start[Get API token]
        Step1{{api_token argument provided?}}
        Step2{{KOLENA_TOKEN environment variable set?}}
        Step3{{Token in .netrc file?}}
        End[Use as API token]
        Exception[MissingTokenError]
        Start --> Step1
        Step1 -->|No| Step2
        Step2 -->|No| Step3
        Step3 -->|No| Exception
        Step1 -->|Yes| End
        Step2 -->|Yes| End
        Step3 -->|Yes| End
    ```

    !!! note
        As of version 0.29.0: the `entity` argument is no longer needed; the signature `initialize(entity, api_token)`
        has been deprecated and replaced by `initialize(api_token)`.

    :param api_token: Directly provide an API token, otherwise attempts to find a token in `$KOLENA_TOKEN`
        or your `.netrc` file. This token is a secret and should be treated with caution.
    :param verbose: Run the client in verbose mode, providing more information about execution. All logging
        events are emitted as Python standard library `logging` events from the `"kolena"` logger as well as
        to stdout/stderr directly.
    :param proxies: Run the client with `http` or `https` proxies. The `proxies` parameter
        is passed through to the `requests` package and can be
        [configured accordingly](https://requests.readthedocs.io/en/latest/user/advanced/#proxies).
    :raises InvalidTokenError: The provided `api_token` is not valid.
    :raises InputValidationError: The provided combination or number of args is not valid.
    :raises MissingTokenError: An API token could not be found.
    """
    log.info("Attempting to initialize client...")
    used_deprecated_signature = "entity" in kwargs

    if len(args) > 2:
        raise InputValidationError(
            f"Too many args. Expected 0..2 but got {len(args)} Check docs for usage.",
        )

    if not api_token:
        if len(args) == 1:
            api_token = args[0]
        elif len(args) == 2:
            api_token = args[1]
            used_deprecated_signature = True
        else:
            api_token = _find_token()

    if used_deprecated_signature:
        warnings.warn(
            "The signature initialize(entity, token) is deprecated.",
            category=DeprecationWarning,
            stacklevel=2,
        )
    assert api_token is not None
    init_response = state.get_token(api_token, proxies=proxies)
    _client_state.update(
        api_token=api_token,
        jwt_token=init_response.access_token,
        tenant=init_response.tenant,
        verbose=verbose,
        proxies=proxies,
    )

    if used_deprecated_signature:
        upload_log("Client attempted to use deprecated entity auth signature.", "warn")

    set_profile()
    record_event(EventAPI.RecordEventRequest(event_name=EventAPI.Event.INITIALIZE_SDK_CLIENT))

    log.info("initialized")
    if verbose:
        # Configure third party logging based on verbosity
        pd.set_option("display.max_colwidth", None)
        log.info(f"connected to {get_platform_url()}")


def _find_token() -> Optional[str]:
    if KOLENA_TOKEN_ENV in os.environ:
        return os.environ[KOLENA_TOKEN_ENV]

    hostname = urlparse(state._get_api_base_url()).hostname or ""
    try:
        netrc_file = netrc.netrc()
        record = netrc_file.authenticators(hostname)
        if record and record[2]:
            return record[2]
        raise MissingTokenError(
            f"No API token in `{KOLENA_TOKEN_ENV}` env variable or in .netrc file under {hostname}",
        )
    except (FileNotFoundError, netrc.NetrcParseError):
        raise MissingTokenError(
            f"No API token in `{KOLENA_TOKEN_ENV}` env variable and unable to parse .netrc file",
        )
