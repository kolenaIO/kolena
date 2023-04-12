from enum import Enum

from pydantic.dataclasses import dataclass


class ClientLog:
    class Path(str, Enum):
        UPLOAD = "/client-log/upload"

    @dataclass(frozen=True)
    class UploadLogRequest:
        client_version: str
        timestamp: str
        message: str
        status: str
