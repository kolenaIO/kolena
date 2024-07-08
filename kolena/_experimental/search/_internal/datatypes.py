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
import pandera as pa
from pandera.typing import Series

# Ensure check method is registered or else would get SchemaInitError
# noreorder
from kolena._utils.dataframes.validators import _validate_locator  # noqa: F401


class LocatorEmbeddingsDataFrameSchema(pa.DataFrameModel):
    key: Series[pa.typing.String] = pa.Field(coerce=True, _validate_locator=())
    """Unique key corresponding to model used for embeddings extraction. This is typically a locator."""

    locator: Series[pa.typing.String] = pa.Field(coerce=True, _validate_locator=())
    """External locator pointing to a sample in bucket."""

    embedding: Series[pa.typing.String] = pa.Field(coerce=True)
    """
    Embedding vector (base64-encoded string of `np.ndarray`) corresponding to a searchable representation of the sample.
    """


class DatasetEmbeddingsDataFrameSchema(pa.DataFrameModel):
    key: Series[pa.typing.String] = pa.Field(coerce=True, _validate_locator=())
    """Unique key corresponding to model used for embeddings extraction. This is typically a locator."""

    datapoint_id_object: Series[pa.typing.String] = pa.Field(coerce=True)
    """
    String representation of the serialized datapoint id object from the dataset's id fields.
    """

    embedding: Series[pa.typing.String] = pa.Field(coerce=True)
    """
    Embedding vector (base64-encoded string of `np.ndarray`) corresponding to a searchable representation of the
        datapoint.
    """
