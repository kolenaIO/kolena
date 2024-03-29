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


@pytest.fixture(scope="session", autouse=True)
def with_init() -> Iterator[None]:
    with kolena_session(api_token=os.environ["KOLENA_TOKEN"]):
        yield


@pytest.fixture(scope="session")
def kolena_token() -> str:
    return os.environ["KOLENA_TOKEN"]


pytest.register_assert_rewrite("tests.integration.helper")
