import typing

import pytest

import kolena.fr
from kolena.errors import DirectInstantiationError
from kolena.fr import Model


@typing.no_type_check
def test_create_model() -> None:
    with pytest.raises(DirectInstantiationError):
        kolena.fr.Model(id=0, name="name", metadata={})

    with pytest.raises(ValueError):
        Model.Data(id=0, name="name", metadata="not a JSON object")

    # expect no error
    kolena.fr.Model.__factory__(Model.Data(id=0, name="name", metadata={}))
