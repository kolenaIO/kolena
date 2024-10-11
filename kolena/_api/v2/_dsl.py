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
from typing import Dict
from typing import Literal

from pydantic import constr
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class ResultContext:
    model_id: int
    eval_config_id: int


Contexts = Dict[str, ResultContext]


@dataclass(frozen=True)
class Dsl:
    contexts: Contexts
    query: constr(min_length=1, strip_whitespace=True)

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.contexts),
                self.query,
            ),
        )


Mode = Literal["sort", "filter", "derived_field"]


@dataclass(frozen=True)
class ValidateDslRequest:
    dsl: Dsl
    mode: Mode
