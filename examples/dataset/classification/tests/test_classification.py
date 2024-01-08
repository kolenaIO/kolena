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
from argparse import Namespace
from collections.abc import Iterator

import pytest
from classification.binary.upload_dataset import run as upload_binary_dataset
from classification.binary.upload_results import run as upload_binary_results
from classification.multiclass.upload_dataset import run as upload_multiclass_dataset
from classification.multiclass.upload_results import run as upload_multiclass_results

from kolena._utils.state import kolena_session

BUCKET = "kolena-public-examples"
BINARY_DATASET = "dogs-vs-cats"
MULTICLASS_DATASET = "cifar10"


@pytest.fixture(scope="session", autouse=True)
def with_init() -> Iterator[None]:
    with kolena_session(api_token=os.environ["KOLENA_TOKEN"]):
        yield


@pytest.fixture(scope="module")
def random_prefix() -> str:
    TEST_PREFIX = "".join(random.choices(string.ascii_uppercase + string.digits, k=12))
    return TEST_PREFIX


def test__upload_dataset(random_prefix: str) -> None:
    args = Namespace(dataset=random_prefix + f" - {BINARY_DATASET}")
    upload_binary_dataset(args)
    args = Namespace(dataset=random_prefix + f" - {MULTICLASS_DATASET}")
    upload_multiclass_dataset(args)


@pytest.mark.depends(on=["test__upload_dataset"])
def test__upload_results(random_prefix: str) -> None:
    args = Namespace(model="inceptionv3", dataset=random_prefix + f" - {BINARY_DATASET}")
    upload_binary_results(args)
    args = Namespace(model="resnet50v2", dataset=random_prefix + f" - {MULTICLASS_DATASET}")
    upload_multiclass_results(args)
