from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import Dict

from kolena._api.v1.detection import Metrics
from kolena._utils.frozen import Frozen
from kolena._utils.serializable import Serializable


class TestConfig(Frozen, Serializable, metaclass=ABCMeta):
    """
    Base class for a testing configuration.

    See concrete implementations :class:`kolena.detection.config.FixedThreshold`,
    :class:`kolena.detection.config.F1Optimal` for details.
    """

    @abstractmethod
    def __init__(self) -> None:
        ...

    @abstractmethod
    def _to_run_config(self) -> Metrics.RunConfig:
        ...

    def _to_dict(self) -> Dict[str, Any]:
        run_config = self._to_run_config()
        return {k: v for k, v in run_config.__dict__.items() if not k.startswith("_")}
