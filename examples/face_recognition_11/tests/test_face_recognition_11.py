# Copyright 2021-2023 Kolena Inc.
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
from face_recognition_11.seed_test_run import main as seed_test_run_main
from face_recognition_11.seed_test_suite import main as seed_test_suite_main


def test__seed_test_suite__smoke() -> None:
    args = Namespace(
        dataset_csv="s3://kolena-public-datasets/labeled-faces-in-the-wild/meta/pairs.sample.csv",
        metadata_csv="s3://kolena-public-datasets/labeled-faces-in-the-wild/meta/metadata.csv",
    )
    seed_test_suite_main(args)


@pytest.mark.depends(on=["test__seed_test_suite__smoke"])
def test__seed_test_run__smoke() -> None:
    args = Namespace(
        models=["Paravision"],
        test_suites=["fr 1:1 :: labeled-faces-in-the-wild"],
    )
    seed_test_run_main(args)
