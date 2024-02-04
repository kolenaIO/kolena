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
from argparse import Namespace

import pytest
from scripts.multiclass.seed_test_run import main as seed_test_run_main
from scripts.multiclass.seed_test_suite import DATASET
from scripts.multiclass.seed_test_suite import main as seed_test_suite_main


@pytest.fixture(scope="module")
def test_suite(test_prefix: str) -> str:
    return f"{test_prefix} - {DATASET}"


def test__seed_test_suite__smoke(test_suite: str) -> None:
    args = Namespace(
        dataset_csv="s3://kolena-public-datasets/cifar10/test/meta/metadata.tiny5.csv",
        test_suite=test_suite,
    )
    seed_test_suite_main(args)


@pytest.mark.depends(on=["test__seed_test_suite__smoke"])
def test__seed_test_run__smoke(test_suite: str) -> None:
    args = Namespace(models=["inceptionv3"], test_suites=[f"image properties :: {test_suite}"])
    seed_test_run_main(args)
