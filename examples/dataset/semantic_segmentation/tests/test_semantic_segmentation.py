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
from semantic_segmentation.constants import BUCKET
from semantic_segmentation.constants import DATASET
from semantic_segmentation.constants import MODEL_NAME
from semantic_segmentation.upload_dataset import run as upload_dataset_run
from semantic_segmentation.upload_results import run as upload_results_run

from kolena._utils.state import kolena_session


@pytest.fixture(scope="session", autouse=True)
def with_init() -> Iterator[None]:
    with kolena_session(api_token=os.environ["KOLENA_TOKEN"]):
        yield


@pytest.fixture(scope="module")
def dataset_name() -> str:
    TEST_PREFIX = "".join(random.choices(string.ascii_uppercase + string.digits, k=12))
    return f"{TEST_PREFIX} - {DATASET}"


def test__semantic_segmentation_upload_dataset__smoke(dataset_name: str) -> None:
    args = Namespace(
        dataset_csv=f"s3://{BUCKET}/{DATASET}/raw/coco-stuff-10k-person.tiny5.csv",
        dataset_name=dataset_name,
    )
    upload_dataset_run(args)


@pytest.mark.depends(on=["test__semantic_segmentation_upload_dataset__smoke"])
def test__semantic_segmentation_upload_results__smoke(dataset_name: str) -> None:
    args = Namespace(
        model=MODEL_NAME["pspnet_r101"],
        write_bucket=BUCKET,
        dataset=dataset_name,
    )
    upload_results_run(args)
