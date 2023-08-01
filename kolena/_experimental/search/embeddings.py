# Copyright 2021-2023 Kolena Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
import json
from typing import List
from typing import Tuple

import numpy.typing as np_typing
import pandas as pd
from dacite import from_dict

from kolena._api.v1.generic import Search as API
from kolena._experimental.search._internal.datatypes import LocatorEmbeddingsDataFrameSchema
from kolena._utils import krequests
from kolena._utils import log
from kolena._utils.batched_load import init_upload
from kolena._utils.batched_load import upload_data_frame
from kolena._utils.consts import BatchSize
from kolena._utils.dataframes.validators import validate_df_schema


def upload_embeddings(embeddings: List[Tuple[str, np_typing.ArrayLike]]) -> None:
    """
    Upload a list of search embeddings corresponding to sample locators.

    :param embeddings: List of locator-embedding pairs, as tuples
    :raises InputValidationError: The provided embeddings input is not of a valid format
    """
    init_response = init_upload()
    locators, search_embeddings = [], []
    for locator, embedding in embeddings:
        locators.append(locator)
        search_embeddings.append(embedding)
    df_embeddings = pd.DataFrame(dict(locator=locators, embedding=search_embeddings))
    df_validated = validate_df_schema(df_embeddings, LocatorEmbeddingsDataFrameSchema)

    upload_data_frame(df=df_validated, batch_size=BatchSize.UPLOAD_EMBEDDINGS.value, load_uuid=init_response.uuid)
    request = API.UploadEmbeddingsRequest(
        uuid=init_response.uuid,
    )
    res = krequests.post(
        endpoint_path=API.Path.EMBEDDINGS.value,
        data=json.dumps(dataclasses.asdict(request)),
    )
    krequests.raise_for_status(res)
    data = from_dict(data_class=API.UploadEmbeddingsResponse, data=res.json())
    log.success(f"uploaded embeddings for {data.n_updated} samples")
