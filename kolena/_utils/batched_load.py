# Copyright 2021-2024 Kolena Inc.
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
import io
import json
import math
import tempfile
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import Union

import numpy as np
import pandas as pd
import requests
from retrying import retry

from kolena._api.v1.batched_load import BatchedLoad as API
from kolena._utils import krequests
from kolena._utils import log
from kolena._utils.datatypes import LoadableDataFrame
from kolena._utils.serde import from_dict
from kolena._utils.state import DEFAULT_API_VERSION

VALIDATION_COUNT_LIMIT = 100
STAGE_STATUS__LOADED = "LOADED"
BATCH_LIMIT = 250 * pow(1024, 2)
SHORT_CIRCUT_LIMIT = 2 * pow(1024, 3)


def init_upload() -> API.InitiateUploadResponse:
    init_res = krequests.put(endpoint_path=API.Path.INIT_UPLOAD.value)
    krequests.raise_for_status(init_res)
    init_response = from_dict(data_class=API.InitiateUploadResponse, data=init_res.json())
    return init_response


def _upload_data_frame(df: pd.DataFrame, batch_size: int, load_uuid: str) -> None:
    num_chunks = math.ceil(len(df) / batch_size)
    chunk_iter = np.array_split(df, num_chunks) if num_chunks > 0 else []

    # only display progress bar if there are multiple chunks to upload
    chunk_iter_logged = log.progress_bar(chunk_iter) if num_chunks > 1 else chunk_iter
    for df_chunk in chunk_iter_logged:
        upload_data_frame_chunk(df_chunk, load_uuid)


@retry(stop_max_attempt_number=3)
def upload_data_frame_chunk(df_chunk: pd.DataFrame, load_uuid: str) -> None:
    # We use a file-like object here so that requests chunks the file upload
    # For reasons not entirely clear, this upload can fail with a broken connection if it is not chunked.
    df_chunk_buffer = io.BytesIO()
    df_chunk.to_parquet(df_chunk_buffer)
    df_chunk_buffer.seek(0)
    signed_url_response = krequests.get(endpoint_path=API.Path.upload_signed_url(load_uuid))
    krequests.raise_for_status(signed_url_response)
    signed_url = from_dict(data_class=API.SignedURL, data=signed_url_response.json())
    upload_response = requests.put(
        url=signed_url.signed_url,
        data=df_chunk_buffer,
        headers={"Content-Type": "application/octet-stream"},
        **krequests.get_connection_args(),
    )
    krequests.raise_for_status(upload_response)


DFType = TypeVar("DFType", bound=LoadableDataFrame)


def _calcuate_batch_size_preflight(df: pd.DataFrame, rows: int, batch_limit: int) -> int:
    return math.ceil((batch_limit / _get_preflight_export_size(df, rows)) * rows)


def _calculate_memory_size(df: pd.DataFrame) -> int:
    return df.memory_usage(index=True, deep=True).sum()


def upload_data_frame(
    df: pd.DataFrame,
    load_uuid: str,
    rows: int = 1000,
    batch_limit: int = BATCH_LIMIT,
    short_circut_limit: int = SHORT_CIRCUT_LIMIT,
) -> None:
    batch_size = (
        len(df)
        if _calculate_memory_size(df) <= short_circut_limit and len(df) > 0
        else _calcuate_batch_size_preflight(df, rows, batch_limit)
    )
    _upload_data_frame(df, batch_size, load_uuid)


def _get_preflight_export_size(df: pd.DataFrame, rows: int) -> int:
    df_subset = df[:rows]
    df_chunk_buffer = io.BytesIO()
    df_subset.to_parquet(df_chunk_buffer)
    file_size = df_chunk_buffer.getbuffer().nbytes
    if not file_size:
        raise ValueError("Exported file has size 0")
    return file_size


class _BatchedLoader(Generic[DFType]):
    @staticmethod
    def load_path(path: str, df_class: Optional[Type[DFType]]) -> Union[DFType, pd.DataFrame]:
        with tempfile.TemporaryFile() as tmp:
            with krequests.get(
                endpoint_path=API.Path.download_by_path(path),
                allow_redirects=True,
                stream=True,
            ) as download_response:
                krequests.raise_for_status(download_response)
                for chunk in download_response.iter_content(chunk_size=8 * 1024**2):
                    tmp.write(chunk)
            tmp.seek(0)
            df = pd.read_parquet(tmp)

        # common postprocessing
        column_mapping = {col_name: col_name.lower() for col_name in df.columns}
        df.rename(columns=column_mapping, inplace=True)

        return df_class.from_serializable(df) if df_class else df

    @staticmethod
    def concat(dfs: Iterable[pd.DataFrame], df_class: Type[DFType]) -> DFType:
        dfs_list = list(dfs)  # collect
        if len(dfs_list) == 0:
            return df_class.construct_empty()
        df = pd.concat(dfs_list)
        df = df.reset_index(drop=True)
        return df_class(df)

    @staticmethod
    def complete_load(uuid: Optional[str], api_version: str = DEFAULT_API_VERSION) -> None:
        if uuid is None:
            return
        complete_request = API.CompleteDownloadRequest(uuid=uuid)
        complete_res = krequests.put(
            endpoint_path=API.Path.COMPLETE_DOWNLOAD.value,
            data=json.dumps(dataclasses.asdict(complete_request)),
            api_version=api_version,
        )
        krequests.raise_for_status(complete_res)

    @staticmethod
    def iter_data(
        init_request: API.BaseInitDownloadRequest,
        endpoint_path: str,
        df_class: Optional[Type[DFType]],
        endpoint_api_version: str = DEFAULT_API_VERSION,
    ) -> Iterator[DFType]:
        with krequests.put(
            endpoint_path=endpoint_path,
            data=json.dumps(dataclasses.asdict(init_request)),
            stream=True,
            api_version=endpoint_api_version,
        ) as init_res:
            krequests.raise_for_status(init_res)
            load_uuid = None
            try:
                for line in init_res.iter_lines():
                    partial_response = from_dict(
                        data_class=API.InitDownloadPartialResponse,
                        data=json.loads(line),
                    )
                    load_uuid = partial_response.uuid
                    yield _BatchedLoader.load_path(partial_response.path, df_class)
            finally:
                _BatchedLoader.complete_load(load_uuid)
