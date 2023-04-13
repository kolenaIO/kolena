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
