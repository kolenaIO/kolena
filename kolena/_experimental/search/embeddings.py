# Copyright 2021-2025 Kolena Inc.
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
import pickle
import warnings
from base64 import b64encode
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from dacite import from_dict

from kolena._api.v1.generic import Search as API
from kolena._experimental.search._internal.datatypes import LocatorEmbeddingsDataFrameSchema
from kolena._utils import krequests
from kolena._utils import log
from kolena._utils.batched_load import init_upload
from kolena._utils.batched_load import upload_data_frame
from kolena._utils.dataframes.validators import validate_df_schema
from kolena.dataset import upload_dataset_embeddings as new_upload_dataset_embeddings
from kolena.errors import InputValidationError


def upload_embeddings(key: str, embeddings: List[Tuple[str, np.ndarray]]) -> None:
    """
    Upload a list of search embeddings corresponding to sample locators.
    :param key: String value uniquely corresponding to the model used to extract the embedding vectors.
        This is typically a locator.
    :param embeddings: List of locator-embedding pairs, as tuples. Locators should be string values, while embeddings
        should be an `numpy.typing.ArrayLike` of numeric values.
    :raises InputValidationError: The provided embeddings input is not of a valid format
    """
    init_response = init_upload()
    locators, search_embeddings = [], []
    for locator, embedding in embeddings:
        if not np.issubdtype(embedding.dtype, np.number):
            raise InputValidationError("unexpected non-numeric embedding dtype")
        locators.append(locator)
        search_embeddings.append(b64encode(pickle.dumps(embedding.astype(np.float32))).decode("utf-8"))
    df_embeddings = pd.DataFrame(dict(key=[key] * len(embeddings), locator=locators, embedding=search_embeddings))
    df_validated = validate_df_schema(df_embeddings, LocatorEmbeddingsDataFrameSchema)

    log.info(f"uploading embeddings for key '{key}'")
    upload_data_frame(df=df_validated, load_uuid=init_response.uuid)
    request = API.UploadEmbeddingsRequest(
        uuid=init_response.uuid,
    )
    res = krequests.post(
        endpoint_path=API.Path.EMBEDDINGS.value,
        data=json.dumps(dataclasses.asdict(request)),
    )
    krequests.raise_for_status(res)
    data = from_dict(data_class=API.UploadEmbeddingsResponse, data=res.json())
    log.success(f"uploaded embeddings for key '{key}' on {data.n_samples} samples")


def upload_dataset_embeddings(dataset_name: str, key: str, df_embedding: pd.DataFrame) -> None:
    """
    Upload a list of search embeddings for a dataset.

    .. deprecated:: 1.55.1
        Use :func:`kolena.dataset.search.upload_dataset_embeddings` instead.

    :param dataset_name: String value indicating the name of the dataset for which the embeddings will be uploaded.
    :param key: String value uniquely corresponding to the embedding vectors. For example, this can be the name of the
        embedding model along with the column with which the embedding was extracted, such as `resnet50-image_locator`.
    :param df_embedding: Dataframe containing id fields for identifying datapoints in the dataset and the associated
        embeddings as `numpy.typing.ArrayLike` of numeric values.
    :raises NotFoundError: The given dataset does not exist.
    :raises InputValidationError: The provided input is not valid.
    """
    warnings.warn(
        "\n kolena._experimental.search.upload_dataset_embeddings is deprecated."
        "\n Use kolena.dataset.search.upload_dataset_embeddings instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    new_upload_dataset_embeddings(dataset_name, key, df_embedding)
