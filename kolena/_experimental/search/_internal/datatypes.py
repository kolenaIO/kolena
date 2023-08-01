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
import numpy as np
import pandera as pa
from pandera.extensions import register_check_method
from pandera.typing import Series

__ALLOWED_INTEGRAL_DTYPES = {
    int,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.int8,
    np.int32,
    np.int64,
    np.dtype("uint8"),
    np.dtype("uint16"),
    np.dtype("uint32"),
    np.dtype("uint64"),
    np.dtype("int8"),
    np.dtype("int32"),
    np.dtype("int64"),
}
__ALLOWED_NUMERIC_DTYPES = {
    *__ALLOWED_INTEGRAL_DTYPES,
    float,
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
    np.dtype("float16"),
    np.dtype("float32"),
    np.dtype("float64"),
    np.dtype("longdouble"),
}


# NOTE: these are actually np.ndarrays, but are declared as objects for compatibility with pandera schemas (as far as
#  pandas is concerned these cells are type 'object')
EmbeddingVector = object  # embedding vector (arbitrary size and data type)


# note that all checks accept nulls -- rely on pandera to properly enforce nullability
def _validate_ndarray(series: Series) -> bool:
    return series.dropna().apply(lambda cell: isinstance(cell, np.ndarray)).all()


@register_check_method()
def _validate_search_embedding_vector(series: Series) -> bool:
    def validate_cell(cell: np.ndarray) -> bool:
        return cell.dtype in __ALLOWED_NUMERIC_DTYPES

    return _validate_ndarray(series) and series.dropna().apply(validate_cell).all()


class LocatorEmbeddingsDataFrameSchema(pa.SchemaModel):
    locator: Series[pa.typing.String] = pa.Field(coerce=True, _validate_locator=())
    """External locator pointing to a sample in bucket."""

    embedding: Series[pa.typing.String] = pa.Field(coerce=True)
    """
    Embedding vector (`np.ndarray`) corresponding to a searchable representation of the sample.
    """
