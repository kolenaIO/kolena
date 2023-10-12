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

    @dataclass(frozen=True)
    class TrackEventRequest:
        event_name: str
        additional_metadata: Optional[Dict[str, str]] = None
