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
#
# flake8: noqa
# autopep8: off
# noreorder
# mypy: ignore-errors
#
# (robots do not like this file)

__name__ = "kolena-client"
__version__: str


def __version_assign() -> None:
    global __version__
    try:
        from importlib.metadata import version

        __version__ = version(__name__)
    except ModuleNotFoundError:
        import importlib_metadata  # importlib.metadata was introduced to the standard library in 3.8

        __version__ = importlib_metadata.version(__name__)


__version_assign()
del __version_assign


import kolena.errors
import kolena.fr
import kolena.detection
import kolena.classification
from .initialize import initialize

__all__ = [
    "initialize",
    "errors",
    "fr",
    "detection",
    "classification",
    "workflow",
]
