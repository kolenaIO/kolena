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
from typing import Dict
from typing import Optional
from typing import Union

from kolena._utils.pydantic_v1 import StrictBool
from kolena._utils.pydantic_v1 import StrictFloat
from kolena._utils.pydantic_v1 import StrictInt
from kolena._utils.pydantic_v1 import StrictStr
from kolena._utils.pydantic_v1.dataclasses import dataclass


class EventAPI:
    class Path(str, Enum):
        PROFILE = "/event/profile"
        EVENT = "/event"

    class Event(str, Enum):
        # auth
        INITIALIZE_SDK_CLIENT = "sdk-client-initialized"

        # test case
        CREATE_TEST_CASE = "sdk-test-case-created"
        LOAD_TEST_CASE = "sdk-test-case-loaded"
        LOAD_TEST_CASE_SAMPLES = "sdk-test-case-samples-loaded"
        EDIT_TEST_CASE = "sdk-test-case-edited"
        INIT_MANY_TEST_CASES = "sdk-many-test-cases-initialized"

        # test suite
        CREATE_TEST_SUITE = "sdk-test-suite-created"
        LOAD_TEST_SUITE = "sdk-test-suite-loaded"
        LOAD_ALL_TEST_SUITES = "sdk-all-test-suites-loaded"
        EDIT_TEST_SUITE = "sdk-test-suite-edited"
        LOAD_TEST_SUITE_SAMPLES = "sdk-test-suite-samples-loaded"

        # model
        CREATE_MODEL = "sdk-model-created"
        LOAD_MODEL = "sdk-model-loaded"
        LOAD_ALL_MODEL = "sdk-all-models-loaded"

        # test run
        EXECUTE_TEST_RUN = "sdk-test-run-executed"

        # dataset
        REGISTER_DATASET = "sdk-dataset-registered"
        FETCH_DATASET = "sdk-dataset-fetched"
        FETCH_DATASET_HISTORY = "sdk-dataset-history-fetched"
        LIST_DATASETS = "sdk-datasets-listed"

        # dataset evaluation
        FETCH_DATASET_MODEL_RESULT = "sdk-dataset-model-result-fetched"
        UPLOAD_DATASET_MODEL_RESULT = "sdk-dataset-model-result-uploaded"

        # quality-standard
        FETCH_QUALITY_STANDARD_RESULT = "sdk-quality-standard-result-fetched"

    @dataclass(frozen=True)
    class RecordEventRequest:
        event_name: str
        additional_metadata: Optional[Dict[str, Union[StrictInt, StrictFloat, StrictStr, StrictBool, None]]] = None
