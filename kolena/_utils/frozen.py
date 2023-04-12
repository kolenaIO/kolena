from contextlib import contextmanager
from typing import Any
from typing import Dict
from typing import Iterator
from typing import TypeVar

from kolena.errors import FrozenObjectError

T = TypeVar("T")


class Frozen:
    _frozen = False

    def _freeze(self) -> None:
        object.__setattr__(self, "_frozen", True)

    @contextmanager
    def _unfrozen(self) -> Iterator[None]:
        object.__setattr__(self, "_frozen", False)
        yield
        self._freeze()

    def _fields(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if not key.startswith("_")}

    def __setattr__(self, key: str, value: T) -> None:
        if self._frozen:
            raise FrozenObjectError("cannot set attribute of frozen object")
        object.__setattr__(self, key, value)

    def __repr__(self) -> str:
        fields = ", ".join(f"{key}={repr(value)}" for key, value in self._fields().items())
        return f"{type(self).__name__}({fields})"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) and self.__dict__ == other.__dict__
