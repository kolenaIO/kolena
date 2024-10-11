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
from typing import Optional

from pydantic import conlist
from pydantic import constr
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class Scopes:
    # None means no scope, empty array means selecting nothing
    datapoint: Optional[conlist(constr(min_length=1))] = None
    result: Optional[conlist(constr(min_length=1))] = None
    human_evaluation: Optional[conlist(constr(min_length=1))] = None

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.datapoint) if self.datapoint is not None else None,
                tuple(self.result) if self.result is not None else None,
                tuple(self.human_evaluation) if self.human_evaluation is not None else None,
            ),
        )
