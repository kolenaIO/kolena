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
from argparse import Namespace
from collections.abc import Iterator

import pytest
from classification.binary.upload_dataset import main as upload_dataset_main
from classification.binary.upload_results import run as upload_results_main

from kolena._utils.state import kolena_session

BUCKET = "kolena-public-datasets"
DATASET = "dogs-vs-cats"


@pytest.fixture(scope="session", autouse=True)
def with_init() -> Iterator[None]:
    with kolena_session(api_token=os.environ["KOLENA_TOKEN"]):
        yield


def test__upload_dataset() -> None:
    upload_dataset_main()


@pytest.mark.depends(on=["test__upload_dataset"])
def test__upload_results() -> None:
    args = Namespace(models=["resnet50v2", "inceptionv3"], multiclass=True)
    upload_results_main(args)
