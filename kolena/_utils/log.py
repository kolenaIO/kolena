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
import logging
import os
import sys
from typing import Any
from typing import Iterable
from typing import Optional
from typing import TypeVar

import termcolor
from tqdm.auto import tqdm

from kolena._utils.state import _client_state


LOGGER_NAME = "kolena"
_logger = logging.getLogger(LOGGER_NAME)
_logger.addHandler(logging.NullHandler())


def _colored(message: str, color: Optional[str] = None) -> str:
    # disable coloration on windows ('nt') as it does not work without hassle
    if color is not None and os.name != "nt" and color in termcolor.COLORS:
        return termcolor.colored(message, color=color)
    else:  # color printing not available on this system
        return message


def _print(message: str, color: Optional[str] = None, **kwargs: Any) -> None:
    print(_colored(message, color), **kwargs)


def _print_header(**kwargs: Any) -> None:
    _print(f"{LOGGER_NAME}> ", color="magenta", end="", **kwargs)


def success(message: str, **kwargs: Any) -> None:
    if _client_state.verbose:
        _print_header()
        _print(message, **kwargs, color="green")
    _logger.info(message, **kwargs)


def info(message: str, **kwargs: Any) -> None:
    if _client_state.verbose:
        _print_header()
        _print(message, **kwargs)
    _logger.info(message, **kwargs)


def warn(message: str, **kwargs: Any) -> None:
    if _client_state.verbose:
        _print_header(file=sys.stderr)
        _print(message, **kwargs, color="yellow", file=sys.stderr)
    _logger.warning(message, **kwargs)


def error(message: str, exception: Optional[BaseException], **kwargs: Any) -> None:
    _print_header(file=sys.stderr)
    _print(message, **kwargs, color="red", file=sys.stderr)
    _logger.error(message, exc_info=exception, **kwargs)


T = TypeVar("T")


def progress_bar(iterator: Iterable[T], desc: Optional[str] = None, **kwargs: Any) -> Iterable[T]:
    if _client_state.verbose:
        desc_base = "kolena> " if is_notebook else _colored("kolena> ", color="magenta")
        desc_full = f"{desc_base}{desc}" if desc is not None else desc_base
        iterator = tqdm(iterator, desc=desc_full, **kwargs)
    yield from iterator


# Logic copied from https://stackoverflow.com/a/39662359
def _is_notebook() -> bool:
    if "IPython" not in sys.modules:
        return False
    try:
        get_ipython = sys.modules["IPython"].get_ipython  # type: ignore
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


is_notebook = _is_notebook()
