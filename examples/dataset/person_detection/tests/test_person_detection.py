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
from person_detection.constants import DATASET_NAME
from person_detection.upload_dataset import run as upload_dataset_main
from person_detection.upload_results import run as upload_results_main

from kolena._utils.state import kolena_session


DATASET = "coco-person-detection"


@pytest.fixture(scope="session", autouse=True)
def with_init() -> Iterator[None]:
    with kolena_session(api_token=os.environ["KOLENA_TOKEN"]):
        yield


@pytest.fixture(scope="module")
def dataset_name() -> str:
    TEST_PREFIX = "".join(random.choices(string.ascii_uppercase + string.digits, k=12))
    return f"{TEST_PREFIX} - {DATASET_NAME}"


def test__upload_dataset(dataset_name: str) -> None:
    args = Namespace(dataset=dataset_name)
    upload_dataset_main(args)


@pytest.mark.depends(on=["test__upload_dataset"])
def test__upload_results(dataset_name: str) -> None:
    args = Namespace(model="yolo_r", dataset=dataset_name)
    upload_results_main(args)
    args = Namespace(model="yolo_x", dataset=dataset_name)
    upload_results_main(args)
    args = Namespace(model="yolo_v3", dataset=dataset_name)
    upload_results_main(args)
    args = Namespace(model="yolo_v4s", dataset=dataset_name)
    upload_results_main(args)
    args = Namespace(model="faster_rcnn", dataset=dataset_name)
    upload_results_main(args)
    args = Namespace(model="mask_rcnn", dataset=dataset_name)
    upload_results_main(args)
