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
        LOAD_ALL_TEST_SUITE = "load-all-test-suite"
        EDIT_TEST_SUITE = "edit-test-suite"
        LOAD_TEST_SUITE_SAMPLES = 'load-test-suite-samples'

    @dataclass(frozen=True)
    class TrackEventRequest:
        event_name: str
        additional_metadata: Optional[Dict[str, str]] = None
