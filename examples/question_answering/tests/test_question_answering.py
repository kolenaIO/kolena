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
from question_answering.seed_test_run import main as seed_test_run_main
from question_answering.seed_test_suite import main as seed_test_suite_main


def test__qa_seed_test_suite__smoke() -> None:
    args = Namespace(dataset_csv="s3://kolena-public-datasets/CoQA/metadata/metadata_head.csv")
    seed_test_suite_main(args)


@pytest.mark.depends(on=["test__qa_seed_test_suite__smoke"])
def test__qa_seed_test_run__smoke() -> None:
    args = Namespace(model="gpt-3.5-turbo_head", test_suite="question types :: CoQA")
    seed_test_run_main(args)
