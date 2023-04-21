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
import concurrent.futures
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Generic
from typing import List
from typing import TypeVar

T = TypeVar("T")


class CombinedFutureError(RuntimeError):
    ...


@dataclass(frozen=True)
class CombinedFuture(Generic[T]):
    futures: List[Future]

    def wait(self) -> List[T]:
        results = []
        for future in concurrent.futures.as_completed(self.futures):
            try:
                results.append(future.result(timeout=1))
            except Exception as e:
                raise CombinedFutureError(f"future failed with error: {type(e).__name__}({e})")
        return results
