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
"""
Reference for various exceptions raised from `kolena`. All custom exceptions extend the base
[`KolenaError`][kolena.errors.KolenaError].
"""
from typing import Any
from typing import Dict
from typing import List

from requests import HTTPError


class KolenaError(Exception):
    """Base error for all Kolena errors to extend. Allows consumers to catch Kolena specific errors."""


class InputValidationError(ValueError, KolenaError):
    """Exception indicating that provided input data failed validation."""


class DuplicateDatapointIdError(InputValidationError):
    """Exception when provided input contains multiple references to the same datapoint id values"""

    def __init__(self, *args: Any, duplicate_ids: List[Dict[str, Any]]) -> None:
        super().__init__(*args)
        self.duplicate_ids = duplicate_ids


class IncorrectUsageError(RuntimeError, KolenaError):
    """Exception indicating that the user performed a disallowed action with the client."""


class InvalidTokenError(ValueError, KolenaError):
    """Exception indicating that provided token value was invalid."""


class InvalidClientStateError(RuntimeError, KolenaError):
    """Exception indicating that client state was invalid."""


class MissingTokenError(KeyError, KolenaError):
    """Exception indicating that the client could not locate an API token."""


class DirectInstantiationError(RuntimeError, KolenaError):
    """
    Exception indicating that the default constructor was used for a class that does not support direct instantiation.
    Available static constructors should be used when this exception is encountered.
    """


class FrozenObjectError(RuntimeError, KolenaError):
    """Exception indicating that the user attempted to modify a frozen object."""


class UnauthenticatedError(HTTPError, KolenaError):
    """Exception indicating unauthenticated usage of the client."""


class RemoteError(HTTPError, KolenaError):
    """Exception indicating that a remote error occurred in communications between the Kolena client and server."""


class CustomMetricsException(KolenaError):
    """Exception indicating that there's an error when computing custom metrics."""


class WorkflowMismatchError(KolenaError):
    """Exception indicating a workflow mismatch."""


class NotFoundError(RemoteError):
    """Exception indicating an entity is not found"""


class NameConflictError(RemoteError):
    """Exception indicating the name of an entity is conflict"""
