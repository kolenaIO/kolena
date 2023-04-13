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
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

import pandas as pd
import pandera as pa
from pandera.extensions import register_check_method
from pandera.typing import Series

from kolena._utils.dataframes.validators import validate_df_schema
from kolena._utils.datatypes import LoadableDataFrame
from kolena._utils.serde import as_deserialized_json
from kolena._utils.serde import as_serialized_json
from kolena._utils.serde import with_serialized_columns
from kolena.detection import Inference


@register_check_method()
def _validate_ground_truths(series: pa.typing.Series) -> bool:
    # expectation: Optional[List[dict(data_type=str, data_object=dict)]]
    def validate_cell(cell: List[Dict[str, Any]]) -> bool:
        return all("data_type" in gt and "data_object" in gt and "label" in gt["data_object"] for gt in cell)

    return series.dropna().apply(validate_cell).all()


JSONObject = object


class TestImageDataFrameSchema(pa.SchemaModel):
    locator: Series[pa.typing.String] = pa.Field(coerce=True, _element_wise_validate_locator=())
    dataset: Series[pa.typing.String] = pa.Field(coerce=True)
    ground_truths: Series[object] = pa.Field(coerce=True, nullable=True, _validate_ground_truths=())
    metadata: Series[JSONObject] = pa.Field(coerce=True)


class ImageResultDataFrameSchema(pa.SchemaModel):
    test_run_id: Series[pa.typing.Int64] = pa.Field(coerce=True)
    model_id: Series[pa.typing.Int64] = pa.Field(coerce=True)
    test_sample_id: Series[pa.typing.Int64] = pa.Field(coerce=True)
    ignore: Series[pa.typing.Bool] = pa.Field(coerce=True)
    data_type: Series[pa.typing.String] = pa.Field(coerce=True, nullable=True)  # null iff no inferences for sample
    label: Series[pa.typing.String] = pa.Field(coerce=True, nullable=True)  # null iff no inferences for sample
    confidence: Series[pa.typing.Float64] = pa.Field(coerce=True, nullable=True)  # null iff no inferences for sample
    # null if no inferences for sample, or if inference is classification data_type
    polygon: Series[JSONObject] = pa.Field(coerce=True, nullable=True)


class TestImageDataFrame(pa.typing.DataFrame[TestImageDataFrameSchema], LoadableDataFrame["TestImageDataFrame"]):
    def as_serializable(self) -> pd.DataFrame:
        object_columns = ["ground_truths", "metadata"]
        return with_serialized_columns(self, object_columns)

    @classmethod
    def get_schema(cls) -> Type[TestImageDataFrameSchema]:
        return TestImageDataFrameSchema

    @classmethod
    def from_serializable(cls, df: pd.DataFrame) -> "TestImageDataFrame":
        df_deserialized = df.copy()
        df_deserialized["ground_truths"] = df["ground_truths"].apply(as_deserialized_json)
        df_deserialized["metadata"] = df["metadata"].apply(as_deserialized_json)
        return cast(TestImageDataFrame, validate_df_schema(df_deserialized, TestImageDataFrameSchema, trusted=True))


class ImageResultDataFrame(pa.typing.DataFrame[ImageResultDataFrameSchema], LoadableDataFrame["ImageResultDataFrame"]):
    def as_serializable(self) -> pd.DataFrame:
        df_serializable = self.copy()
        df_serializable["polygon"] = df_serializable["polygon"].apply(as_serialized_json)
        return df_serializable

    @classmethod
    def get_schema(cls) -> Type[ImageResultDataFrameSchema]:
        return ImageResultDataFrameSchema

    @classmethod
    def from_image_inference_mapping(
        cls,
        test_run_id: int,
        model_id: int,
        image_inferences: Dict[int, List[Optional[Inference]]],
        ignored_image_ids: List[int],
    ) -> "ImageResultDataFrame":
        records: List[Tuple[int, bool, Optional[str], Optional[str], Optional[float], Optional[dict]]] = []
        for image_id, inferences in image_inferences.items():
            for inference in inferences:
                if inference is None:
                    records.append((image_id, False, None, None, None, None))
                    continue
                inference_dict = inference._to_dict()
                records.append(
                    (
                        image_id,
                        False,
                        inference_dict["data_type"],
                        inference_dict["data_object"]["label"],
                        inference_dict["data_object"]["confidence"],
                        inference_dict["data_object"].get("points", None),
                    ),
                )
        for image_id in ignored_image_ids:
            records.append((image_id, True, None, None, None, None))
        df_stage = pd.DataFrame(
            records,
            columns=["test_sample_id", "ignore", "data_type", "label", "confidence", "polygon"],
        )
        df_stage["test_run_id"] = test_run_id
        df_stage["model_id"] = model_id
        return ImageResultDataFrame(validate_df_schema(df_stage, ImageResultDataFrameSchema, trusted=True))


class LoadInferencesDataFrameSchema(TestImageDataFrameSchema):
    test_case_id: Series[pa.typing.Int64] = pa.Field(coerce=True)
    inferences: Series[JSONObject] = pa.Field(coerce=True, nullable=True)  # null if ignored


class LoadInferencesDataFrame(
    pa.typing.DataFrame[LoadInferencesDataFrameSchema],
    LoadableDataFrame["LoadInferencesDataFrame"],
):
    def as_serializable(self) -> pd.DataFrame:
        object_columns = ["ground_truths", "metadata", "inferences"]
        return with_serialized_columns(self, object_columns)

    @classmethod
    def get_schema(cls) -> Type[LoadInferencesDataFrameSchema]:
        return LoadInferencesDataFrameSchema

    @classmethod
    def from_serializable(cls, df: pd.DataFrame) -> "LoadInferencesDataFrame":
        df_deserialized = df.copy()
        df_deserialized["ground_truths"] = df["ground_truths"].apply(as_deserialized_json)
        df_deserialized["metadata"] = df["metadata"].apply(as_deserialized_json)
        df_deserialized["inferences"] = df["inferences"].apply(as_deserialized_json)
        return cast(
            LoadInferencesDataFrame,
            validate_df_schema(df_deserialized, LoadInferencesDataFrameSchema, trusted=True),
        )
