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
