import pytest

from kolena._utils.frozen import Frozen
from kolena.errors import FrozenObjectError


class TestFrozen(Frozen):
    def __init__(self) -> None:
        self.a = "a"
        self._freeze()


def test_frozen() -> None:
    obj = TestFrozen()
    with pytest.raises(FrozenObjectError):
        obj.a = "b"

    assert obj.a == "a"


def test_unfrozen() -> None:
    obj = TestFrozen()
    with obj._unfrozen():
        obj.a = "new"

    assert obj.a == "new"

    with pytest.raises(FrozenObjectError):
        obj.a = "b"
