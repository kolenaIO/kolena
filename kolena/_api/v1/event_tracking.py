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

    @dataclass(frozen=True)
    class TrackEventRequest:
        event_name: str
        additional_metadata: Optional[Dict[str, str]] = None
