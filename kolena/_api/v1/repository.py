from enum import Enum

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class CreateRepositoryRequest:
    repository: str


class Path(str, Enum):
    CREATE = "/ecr/repository"
