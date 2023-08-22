from dataclasses import dataclass
from uuid import uuid4

def _random_name() -> str:
    return str(uuid4())

@dataclass
class DataIngestionConfig:
    data_path: str
    locator_prefix: str = ""
    test_case_name: str = _random_name()
    test_suite_name: str = _random_name()
    reset: bool = False
