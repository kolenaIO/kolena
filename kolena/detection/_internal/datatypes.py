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
import json
from typing import cast
from typing import Type

import pandas as pd
import pandera as pa
from pandera.typing import Series

from kolena._utils.dataframes.validators import validate_df_schema
from kolena._utils.datatypes import LoadableDataFrame

JSONObject = object


class LoadTestImagesDataFrameSchema(pa.SchemaModel):
    test_sample_id: Series[pa.typing.Int64] = pa.Field(coerce=True)
    locator: Series[pa.typing.String] = pa.Field(coerce=True, _validate_locator=())
    dataset: Series[pa.typing.String] = pa.Field(coerce=True, nullable=True)
    metadata: Series[JSONObject] = pa.Field(coerce=True)


class LoadTestImagesDataFrame(
    pa.typing.DataFrame[LoadTestImagesDataFrameSchema],
    LoadableDataFrame["LoadTestImagesDataFrame"],
):
    @classmethod
    def get_schema(cls) -> Type[LoadTestImagesDataFrameSchema]:
        return LoadTestImagesDataFrameSchema

    @classmethod
    def from_serializable(cls, df_samples: pd.DataFrame) -> "LoadTestImagesDataFrame":
        df = df_samples.copy()
        df["locator"] = df["data"].apply(lambda v: json.loads(v)["locator"])
        df["metadata"] = df["metadata"].apply(json.loads)
        df = df[["test_sample_id", "locator", "dataset", "metadata"]]
        return cast(LoadTestImagesDataFrame, validate_df_schema(df, LoadTestImagesDataFrameSchema, trusted=True))
