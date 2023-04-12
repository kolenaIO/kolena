from enum import Enum


class _BatchSize(int, Enum):
    UPLOAD_CHIPS = 5_000
    UPLOAD_RECORDS = 10_000_000
    UPLOAD_RESULTS = 1_000_000

    LOAD_RECORDS = UPLOAD_RECORDS
    LOAD_SAMPLES = 1_000_000
