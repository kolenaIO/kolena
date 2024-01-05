import random
import string

import pytest
from scripts.binary.seed_test_suite import DATASET

@pytest.fixture(scope="module")
def suite_name() -> str:
    TEST_PREFIX = "".join(random.choices(string.ascii_uppercase + string.digits, k=12))
    return f"{TEST_PREFIX} - {DATASET}"
