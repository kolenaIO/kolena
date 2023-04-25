import dataclasses
import sys

import pydantic
import pytest

from kolena.workflow._datatypes import DataObject


# extensions must be frozen
@pytest.mark.skipif(sys.version_info < (3, 8), reason="Constraint not enforced on Python < 3.8")
def test_frozen() -> None:
    with pytest.raises(TypeError):

        @dataclasses.dataclass
        class DataclassesTester(DataObject):
            ...

    with pytest.raises(TypeError):

        @pydantic.dataclasses.dataclass
        class PydanticTester(DataObject):
            ...


# can use either stdlib dataclasses or pydantic dataclasses interchangeably
def test_dataclasses_or_pydantic() -> None:
    @dataclasses.dataclass(frozen=True)
    class DataclassesTester(DataObject):
        ...

    DataclassesTester()

    @pydantic.dataclasses.dataclass(frozen=True)
    class PydanticTester(DataObject):
        ...

    PydanticTester()


def test_dataclasses_serialize_order() -> None:
    @dataclasses.dataclass(frozen=True)
    class DataclassesTester(DataObject):
        b: float
        a: str
        z: bool

    tester = DataclassesTester(z=False, a="foobar", b=0.3)
    serialized = tester._to_dict()
    assert list(serialized.keys()) == ["b", "a", "z"]
