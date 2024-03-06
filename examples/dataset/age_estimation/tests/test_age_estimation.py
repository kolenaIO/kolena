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
from age_estimation.upload_dataset import run as upload_dataset_main
from age_estimation.upload_results import run as upload_results_main


BUCKET = "kolena-public-examples"
DATASET = "labeled-faces-in-the-wild"

TEST_PREFIX = "".join(random.choices(string.ascii_uppercase + string.digits, k=12))


@pytest.fixture(scope="module")
def dataset_name() -> str:
    TEST_PREFIX = "".join(random.choices(string.ascii_uppercase + string.digits, k=12))
    return f"{TEST_PREFIX} - {DATASET}"


def test__upload_dataset(dataset_name: str) -> None:
    args = Namespace(dataset=dataset_name)
    upload_dataset_main(args)


@pytest.mark.depends(on=["test__upload_dataset"])
def test__upload_results_deepface(dataset_name: str) -> None:
    args = Namespace(dataset=dataset_name, model="deepface")
    upload_results_main(args)


@pytest.mark.depends(on=["test__upload_dataset"])
def test__upload_results_ssrnet(dataset_name: str) -> None:
    args = Namespace(dataset=dataset_name, model="ssrnet")
    upload_results_main(args)
