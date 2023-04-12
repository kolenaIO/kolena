import json
from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import Dict


class Serializable(metaclass=ABCMeta):
    @abstractmethod
    def _to_dict(self) -> Dict[str, Any]:
        ...

    def __hash__(self) -> int:
        return hash(json.dumps(self._to_dict()))
