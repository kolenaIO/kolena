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
import io
import math

import numpy as np
from PIL import Image
from requests_toolbelt import MultipartEncoder

from kolena._api.v1.fr import Asset as AssetAPI
from kolena._utils import krequests
from kolena._utils.asset_path_mapper import AssetPathMapper
from kolena._utils.consts import BatchSize
from kolena.fr.datatypes import _ImageChipsDataFrame


def upload_image_chips(df: _ImageChipsDataFrame, batch_size: int = BatchSize.UPLOAD_CHIPS.value) -> None:
    def upload_batch(df_batch: _ImageChipsDataFrame) -> None:
        df_batch = df_batch.reset_index(drop=True)  # reset indices so we match the signed_url indices

        def as_buffer(image_raw: np.ndarray) -> io.BytesIO:
            pil_image = Image.fromarray(image_raw).convert("RGB")
            buf = io.BytesIO()
            pil_image.save(buf, "png")
            buf.seek(0)
            return buf

        fields = [
            (
                "files",
                AssetPathMapper.path_stub(row["test_run_id"], row["uuid"], row["image_id"], row["key"]),
                as_buffer(row["image"]),
            )
            for _, row in df_batch.iterrows()
        ]
        data = MultipartEncoder(fields=fields)
        upload_response = krequests.put(
            endpoint_path=AssetAPI.Path.BULK_UPLOAD.value,
            data=data,
            headers={"Content-Type": data.content_type},
        )
        krequests.raise_for_status(upload_response)

    num_chunks = math.ceil(len(df) / batch_size)
    chunk_iter = np.array_split(df, num_chunks) if len(df) > 0 else []
    for df_chunk in chunk_iter:
        upload_batch(df_chunk)
