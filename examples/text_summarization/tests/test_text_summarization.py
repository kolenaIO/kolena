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
import os
from argparse import Namespace
from collections.abc import Iterator

import pytest
from text_summarization.seed_test_run import run as seed_test_run_main
from text_summarization.seed_test_suite import run as seed_test_suite_main

from kolena._utils.state import kolena_session


@pytest.fixture(scope="session", autouse=True)
def with_init() -> Iterator[None]:
    with kolena_session(api_token=os.environ["KOLENA_TOKEN"]):
        yield


def test__seed_test_suite() -> None:
    args = Namespace(dataset_csv="s3://kolena-public-datasets/CNN-DailyMail/metadata/metadata.tiny1.csv")
    seed_test_suite_main(args)


@pytest.mark.depends(on=["test__seed_test_suite"])
def test__seed_test_run() -> None:
    args = Namespace(model="ada", test_suite="CNN-DailyMail :: text length", local_csv=None)
    seed_test_run_main(args)
