from enum import Enum


class _BatchSize(int, Enum):
    UPLOAD_CHIPS = 5_000
    UPLOAD_RECORDS = 10_000_000
