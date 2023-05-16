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
import typing

import pytest

import kolena.fr
from kolena.errors import DirectInstantiationError
from kolena.fr import Model


@typing.no_type_check
def test__model__create() -> None:
    with pytest.raises(DirectInstantiationError):
        kolena.fr.Model(id=0, name="name", metadata={})

    with pytest.raises(ValueError):
        Model.Data(id=0, name="name", metadata="not a JSON object")

    # expect no error
    kolena.fr.Model.__factory__(Model.Data(id=0, name="name", metadata={}))
