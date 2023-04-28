import os
import random
import string
from typing import Iterator

import pytest

from kolena._utils.state import kolena_session

TEST_PREFIX = "".join(random.choices(string.ascii_uppercase + string.digits, k=12))


@pytest.fixture(scope="session", autouse=True)
def log_test_prefix():
    print(f"Using test prefix '{TEST_PREFIX}'")


@pytest.fixture(scope="session")
def with_init() -> Iterator[None]:
    with kolena_session(api_token=os.environ["KOLENA_TOKEN"]):
        yield


pytest.register_assert_rewrite("tests.integration.helper")
