from enum import Enum


class WorkflowType(str, Enum):
    FR = "FR"
    CLASSIFICATION = "CLASSIFICATION"
    DETECTION = "DETECTION"
    UNKNOWN = "UNKNOWN"
