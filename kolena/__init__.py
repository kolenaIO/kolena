# robots do not like this file
#
# flake8: noqa
# autopep8: off
# noreorder
# mypy: ignore-errors

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
