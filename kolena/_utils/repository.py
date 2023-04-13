import dataclasses
import json

from kolena._api.v1.repository import CreateRepositoryRequest
from kolena._api.v1.repository import Path
from kolena._utils import krequests


def create(repository: str) -> None:
    response = krequests.post(
        endpoint_path=Path.CREATE,
        data=json.dumps(dataclasses.asdict(CreateRepositoryRequest(repository=repository))),
    )
    krequests.raise_for_status(response)
