# Copyright 2021-2025 Kolena Inc.
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
import uuid
from argparse import Namespace
from typing import List

import pytest
from personal_data_detection.upload_dataset import run as upload_dataset_main

from kolena.dataset import list_datasets


@pytest.mark.parametrize(
    "allowed_pii_types, should_upload",
    [
        ([], False),
        (["I-USERNAME"], False),
        (["I-USERNAME", "I-GIVENNAME"], False),
        (["I-GIVENNAME", "I-SURNAME"], False),
        (["I-USERNAME", "I-GIVENNAME", "I-SURNAME"], True),
        (["I-USERNAME", "I-GIVENNAME", "I-SURNAME", "I-TELEPHONENUM"], True),
    ],
)
def test__upload_dataset(allowed_pii_types: List[str], should_upload: bool) -> None:
    dataset_name = str(uuid.uuid4())
    args = Namespace(dataset=dataset_name, allowed_pii_types=allowed_pii_types)
    upload_dataset_main(args)

    existing_datasets = list_datasets()
    assert (dataset_name in existing_datasets) == should_upload
