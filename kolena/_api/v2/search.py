from pydantic.dataclasses import dataclass

from enum import Enum

from kolena._api.v1.batched_load import BatchedLoad


class Path(str, Enum):
    EMBEDDINGS = "/search/embeddings"


@dataclass(frozen=True)
class UploadDatasetEmbeddingsRequest(BatchedLoad.WithLoadUUID):
    name: str


@dataclass(frozen=True)
class UploadDatasetEmbeddingsResponse:
    n_datapoints: int
