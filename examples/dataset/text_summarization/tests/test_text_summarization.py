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
from argparse import Namespace

import pytest
from text_summarization.constants import BUCKET
from text_summarization.constants import DATASET
from text_summarization.constants import MODELS
from text_summarization.upload_dataset import main as upload_dataset_main
from text_summarization.upload_results import main as upload_results_main


def test__upload_dataset__smoke() -> None:
    args = Namespace(dataset_csv=f"s3://{BUCKET}/{DATASET}/CNN-DailyMail.tiny1.csv")
    upload_dataset_main(args)


@pytest.mark.depends(on=["test__upload_dataset__smoke"])
def test__upload_results__smoke() -> None:
    args = Namespace(models=[MODELS[0]])
    upload_results_main(args)
