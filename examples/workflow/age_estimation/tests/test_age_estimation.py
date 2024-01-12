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
import random
import string
from argparse import Namespace

import pytest
from age_estimation.seed_test_run import main as seed_test_run_main
from age_estimation.seed_test_suite import DATASET
from age_estimation.seed_test_suite import main as seed_test_suite_main


@pytest.fixture(scope="module")
def test_suite() -> str:
    TEST_PREFIX = "".join(random.choices(string.ascii_uppercase + string.digits, k=12))
    return f"{TEST_PREFIX} - {DATASET}"


def test__seed_test_suite__smoke(test_suite: str) -> None:
    args = Namespace(
        dataset_csv="s3://kolena-public-datasets/labeled-faces-in-the-wild/meta/metadata.tiny5.csv",
        test_suite=test_suite,
    )
    seed_test_suite_main(args)


@pytest.mark.depends(on=["test__seed_test_suite__smoke"])
def test__seed_test_run__smoke(test_suite: str) -> None:
    args = Namespace(model="ssrnet", test_suites=[f"age :: {test_suite} [age estimation]"])
    seed_test_run_main(args)