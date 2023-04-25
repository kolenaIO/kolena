from typing import Any
from typing import Iterable

from tests.integration.conftest import TEST_PREFIX


def fake_locator(index: int, directory: str = "default") -> str:
    return f"https://fake-locator/{directory}/{index}.png"


def with_test_prefix(value: str) -> str:
    return f"{TEST_PREFIX} {value}"


def assert_sorted_list_equal(list_a: Iterable[Any], list_b: Iterable[Any]) -> None:
    assert sorted(list_a) == sorted(list_b)
