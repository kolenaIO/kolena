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
from base64 import b64encode
from typing import Any
from typing import Set

import numpy as np
import pandas as pd
import pandera as pa
from dacite import from_dict
from pandera.typing import Series

from kolena._api.v1.event import EventAPI
from kolena._api.v2.search import Path as PATH_V2
from kolena._api.v2.search import UploadDatasetEmbeddingsRequest
from kolena._api.v2.search import UploadDatasetEmbeddingsResponse
from kolena._utils import krequests
from kolena._utils import log
from kolena._utils.batched_load import init_upload
from kolena._utils.batched_load import upload_data_frame
from kolena._utils.dataframes.validators import validate_df_schema
from kolena._utils.instrumentation import with_event
from kolena._utils.state import API_V2
from kolena.dataset._common import COL_DATAPOINT_ID_OBJECT
from kolena.dataset._common import validate_dataframe_ids
from kolena.dataset.dataset import _load_dataset_metadata
from kolena.dataset.dataset import _to_serialized_dataframe
from kolena.errors import InputValidationError

# Ensure check method is registered or else would get SchemaInitError
# noreorder
from kolena._utils.dataframes.validators import _validate_locator  # noqa: F401


class DatasetEmbeddingsDataFrameSchema(pa.DataFrameModel):
    key: Series[pa.typing.String] = pa.Field(coerce=True)
    """
    Unique key corresponding  to the embedding vectors. This can be, for example, the name of the embedding model along
    with the column with which the embedding was extracted, such as "resnet50-image_locator".
    """

    datapoint_id_object: Series[pa.typing.String] = pa.Field(coerce=True)
    """
    String representation of the serialized datapoint id object from the dataset's id fields.
    """

    embedding: Series[pa.typing.String] = pa.Field(coerce=True)
    """
    Embedding vector (base64-encoded string of `np.ndarray`) corresponding to a searchable representation of the
        datapoint.
    """


def _upload_dataset_embeddings(
    dataset_name: str,
    key: str,
    df_embedding: pd.DataFrame,
    run_embedding_reduction_pipeline: bool = True,
) -> None:
    dataset_entity_data = _load_dataset_metadata(dataset_name)
    assert dataset_entity_data
    embedding_lengths: Set[int] = set()

    def encode_embedding(embedding: Any) -> str:
        if not np.issubdtype(embedding.dtype, np.number):
            raise InputValidationError("unexpected non-numeric embedding dtype")
        embedding_lengths.add(len(embedding))
        return b64encode(pickle.dumps(embedding.astype(np.float32))).decode("utf-8")

    # encode embeddings to string
    df_embedding["embedding"] = df_embedding["embedding"].apply(encode_embedding)
    if len(embedding_lengths) > 1:
        raise InputValidationError(f"embeddings are not of the same size, found {embedding_lengths}")

    id_fields = dataset_entity_data.id_fields
    dataset_name = dataset_entity_data.name
    validate_dataframe_ids(df_embedding, id_fields)
    df_serialized_datapoint_id_object = _to_serialized_dataframe(
        df_embedding[sorted(id_fields)],
        column=COL_DATAPOINT_ID_OBJECT,
    )
    df_embedding = pd.concat([df_embedding, df_serialized_datapoint_id_object], axis=1)

    df_embedding["key"] = key
    df_embedding = df_embedding[[COL_DATAPOINT_ID_OBJECT, "key", "embedding"]]
    df_validated = validate_df_schema(df_embedding, DatasetEmbeddingsDataFrameSchema)

    log.info(f"uploading embeddings for dataset '{dataset_name}' and key '{key}'")
    init_response = init_upload()
    upload_data_frame(df=df_validated, load_uuid=init_response.uuid)
    request = UploadDatasetEmbeddingsRequest(
        uuid=init_response.uuid,
        name=dataset_name,
        run_embedding_reduction=run_embedding_reduction_pipeline,
    )
    res = krequests.post(
        endpoint_path=PATH_V2.EMBEDDINGS.value,
        api_version=API_V2,
        data=json.dumps(dataclasses.asdict(request)),
    )
    krequests.raise_for_status(res)
    data = from_dict(data_class=UploadDatasetEmbeddingsResponse, data=res.json())
    log.success(f"uploaded embeddings for dataset '{dataset_name}' and key '{key}' on {data.n_datapoints} datapoints")


@with_event(event_name=EventAPI.Event.UPLOAD_DATASET_EMBEDDINGS)
def upload_dataset_embeddings(dataset_name: str, key: str, df_embedding: pd.DataFrame) -> None:
    """
    Upload a list of search embeddings for a dataset.

    :param dataset_name: String value indicating the name of the dataset for which the embeddings will be uploaded.
    :param key: String value uniquely corresponding to the embedding vectors. For example, this can be the name of the
        embedding model along with the column with which the embedding was extracted, such as `resnet50-image_locator`.
    :param df_embedding: Dataframe containing id fields for identifying datapoints in the dataset and the associated
        embeddings as `numpy.typing.ArrayLike` of numeric values.
    :raises NotFoundError: The given dataset does not exist.
    :raises InputValidationError: The provided input is not valid.
    """
    _upload_dataset_embeddings(dataset_name, key, df_embedding)
