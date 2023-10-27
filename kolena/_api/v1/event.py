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
from enum import Enum
from typing import Dict
from typing import Optional
from typing import Union

from pydantic.dataclasses import dataclass
from pydantic.types import StrictBool
from pydantic.types import StrictFloat
from pydantic.types import StrictInt
from pydantic.types import StrictStr


class EventAPI:
    class Path(str, Enum):
        PROFILE = "/event/profile"
        EVENT = "/event"

    class Event(str, Enum):
        # auth
        INITIALIZE_SDK_CLIENT = "sdk-client-initialized"

        # test case
        CREATE_TEST_CASE = "test-case-created"
        LOAD_TEST_CASE = "test-case-loaded"
        LOAD_TEST_CASE_SAMPLES = "test-case-samples-loaded"
        EDIT_TEST_CASE = "test-case-edited"
        INIT_MANY_TEST_CASES = "many-test-cases-initialized"

        # test suite
        CREATE_TEST_SUITE = "test-suite-created"
        LOAD_TEST_SUITE = "test-suite-loaded"
        LOAD_ALL_TEST_SUITES = "all-test-suites-loaded"
        EDIT_TEST_SUITE = "test-suite-edited"
        LOAD_TEST_SUITE_SAMPLES = "test-suite-samples-loaded"

        # model
        CREATE_MODEL = "model-created"
        LOAD_MODEL = "model-loaded"

        # test run
        EXECUTE_TEST_RUN = "test-run-executed"

    @dataclass(frozen=True)
    class RecordEventRequest:
        event_name: str
        additional_metadata: Optional[Dict[str, Union[StrictInt, StrictFloat, StrictStr, StrictBool, None]]] = None
