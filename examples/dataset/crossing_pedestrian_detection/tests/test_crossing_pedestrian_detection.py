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
from crossing_pedestrian_detection.constants import DATASET
from crossing_pedestrian_detection.upload_dataset import run as upload_dataset_run
from crossing_pedestrian_detection.upload_results import run as upload_results_run


@pytest.fixture(scope="module")
def dataset_name() -> str:
    TEST_PREFIX = "".join(random.choices(string.ascii_uppercase + string.digits, k=12))
    return f"{TEST_PREFIX} - {DATASET}"


def test__crossing_pedestrian_detection_upload_dataset__smoke(dataset_name: str) -> None:
    args = Namespace(
        dataset=dataset_name,
    )
    upload_dataset_run(args)


@pytest.mark.depends(on=["test__crossing_pedestrian_detection_upload_dataset__smoke"])
def test__crossing_pedestrian_detection_upload_results__smoke(dataset_name: str) -> None:
    args = Namespace(
        model="c3d_sort",
        dataset=dataset_name,
        sample_count=50,
    )
    upload_results_run(args)
