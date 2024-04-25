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
from enum import Enum

KOLENA_TOKEN_ENV = "KOLENA_TOKEN"


class BatchSize(int, Enum):
    UPLOAD_RECORDS = 10_000_000
    UPLOAD_EMBEDDINGS = 1_000_000

    LOAD_RECORDS = UPLOAD_RECORDS
    LOAD_SAMPLES = 1_000_000


class FieldName(str, Enum):
    TEST_CASE_NAME = "Test Case name"
    TEST_SUITE_NAME = "Test Suite name"
    MODEL_NAME = "Model name"
