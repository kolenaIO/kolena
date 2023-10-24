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
from dataclasses import dataclass
from enum import Enum
from typing import Dict
from typing import Optional


class Tracking:
    class Path(str, Enum):
        PROFILE = "/tracking/profile"
        EVENT = "/tracking/event"

    class Events(str, Enum):
        # auth
        GENERATE_TOKEN = "generate-token"

        # test case
        CREATE_TEST_CASE = "create-test-case"
        LOAD_TEST_CASE = "load-test-case"
        LOAD_TEST_CASE_SAMPLES = "load-test-case-samples"
        EDIT_TEST_CASE = "edit-test-case"
        INIT_MANY_TEST_CASES = "initialize-many-test-cases"

        # test suite
        CREATE_TEST_SUITE = "create-test-suite"
        LOAD_TEST_SUITE = "load-test-suite"
        LOAD_ALL_TEST_SUITES = "load-all-test-suites"
        EDIT_TEST_SUITE = "edit-test-suite"
        LOAD_TEST_SUITE_SAMPLES = "load-test-suite-samples"

        # model
        CREATE_MODEL = "create-model"
        LOAD_MODEL = "load-model"

        # test run
        EXECUTE_TEST_RUN = "execute-test-run"

    @dataclass(frozen=True)
    class TrackEventRequest:
        event_name: str
        additional_metadata: Optional[Dict[str, str]] = None
