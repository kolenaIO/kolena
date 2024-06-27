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
from typing import cast
from typing import Optional
from typing import Type

import pandas as pd
import pandera as pa
from pandera.typing import Series

from kolena._utils.dataframes.validators import validate_df_schema
from kolena._utils.datatypes import LoadableDataFrame
from kolena._utils.serde import as_deserialized_json
from kolena._utils.serde import as_serialized_json
from kolena._utils.serde import with_serialized_columns


_SCALAR_TYPES = [str, bool, int, float]
JSONObject = object


class TestSampleDataFrameSchema(pa.DataFrameModel):
    """General-purpose frame used for test samples in isolation or paired with ground truths and/or inferences."""

    test_sample: Series[JSONObject] = pa.Field(coerce=True)
    test_sample_metadata: Optional[Series[JSONObject]] = pa.Field(coerce=True)
    ground_truth: Optional[Series[JSONObject]] = pa.Field(coerce=True, nullable=True)
    inference: Optional[Series[JSONObject]] = pa.Field(coerce=True)


class TestSampleDataFrame(LoadableDataFrame[TestSampleDataFrameSchema]):
    @classmethod
    def get_schema(cls) -> Type[TestSampleDataFrameSchema]:
        return TestSampleDataFrameSchema

    def as_serializable(self) -> pd.DataFrame:
        return TestSampleDataFrame._serde(self, True)

    @classmethod
    def from_serializable(cls, df: pd.DataFrame) -> "TestSampleDataFrame":
        df_deserialized = TestSampleDataFrame._serde(df, False)
        df_validated = validate_df_schema(df_deserialized, TestSampleDataFrameSchema, trusted=True)
        return cast(TestSampleDataFrame, df_validated)

    @staticmethod
    def _serde(df: pd.DataFrame, serialize: bool) -> pd.DataFrame:
        serde_function = as_serialized_json if serialize else as_deserialized_json
        df_out = df.copy()
        df_out["test_sample"] = df_out["test_sample"].apply(serde_function)
        if "test_sample_metadata" in df.columns:
            df_out["test_sample_metadata"] = df_out["test_sample_metadata"].apply(serde_function)
        if "ground_truth" in df.columns:
            df_out["ground_truth"] = df_out["ground_truth"].apply(serde_function)
        if "inference" in df.columns:
            df_out["inference"] = df_out["inference"].apply(serde_function)
        return df_out


class TestSuiteTestSamplesDataFrameSchema(pa.DataFrameModel):
    """Data frame used for loading test samples grouped by test case."""

    test_case_id: Optional[Series[pa.typing.Int64]] = pa.Field(coerce=True)
    test_sample: Series[JSONObject] = pa.Field(coerce=True)
    test_sample_metadata: Series[JSONObject] = pa.Field(coerce=True)


class TestSuiteTestSamplesDataFrame(LoadableDataFrame[TestSuiteTestSamplesDataFrameSchema]):
    @classmethod
    def get_schema(cls) -> Type[TestSuiteTestSamplesDataFrameSchema]:
        return TestSuiteTestSamplesDataFrameSchema

    def as_serializable(self) -> pd.DataFrame:
        return TestSuiteTestSamplesDataFrame._serde(self, True)

    @classmethod
    def from_serializable(cls, df: pd.DataFrame) -> "TestSuiteTestSamplesDataFrame":
        df_deserialized = TestSuiteTestSamplesDataFrame._serde(df, False)
        df_validated = validate_df_schema(df_deserialized, TestSuiteTestSamplesDataFrameSchema, trusted=True)
        return cast(TestSuiteTestSamplesDataFrame, df_validated)

    @staticmethod
    def _serde(df: pd.DataFrame, serialize: bool) -> pd.DataFrame:
        serde_function = as_serialized_json if serialize else as_deserialized_json
        df_out = df.copy()
        df_out["test_sample"] = df_out["test_sample"].apply(serde_function)
        df_out["test_sample_metadata"] = df_out["test_sample_metadata"].apply(serde_function)
        return df_out


class TestCaseEditorDataFrameSchema(pa.DataFrameModel):
    test_case_name: Series[pa.typing.String] = pa.Field(nullable=True)
    test_sample_type: Series[pa.typing.String] = pa.Field(coerce=True)
    test_sample: Series[JSONObject] = pa.Field(coerce=True)  # TODO: validators?
    test_sample_metadata: Series[JSONObject] = pa.Field(coerce=True)
    ground_truth: Series[JSONObject] = pa.Field(coerce=True, nullable=True)
    remove: Series[pa.typing.Bool] = pa.Field(coerce=True)


class TestCaseEditorDataFrame(LoadableDataFrame[TestCaseEditorDataFrameSchema]):
    def as_serializable(self) -> pd.DataFrame:
        object_columns = ["test_sample", "test_sample_metadata", "ground_truth"]
        return with_serialized_columns(self, object_columns)

    @classmethod
    def get_schema(cls) -> Type[TestCaseEditorDataFrameSchema]:
        return TestCaseEditorDataFrameSchema

    @classmethod
    def from_serializable(cls, df: pd.DataFrame) -> "TestCaseEditorDataFrame":
        df_deserialized = df.copy()
        df_deserialized["test_sample"] = df["test_sample"].apply(as_deserialized_json)
        df_deserialized["test_sample_metadata"] = df["test_sample_metadata"].apply(as_deserialized_json)
        df_deserialized["ground_truths"] = df["ground_truths"].apply(as_deserialized_json)
        df_validated = validate_df_schema(df_deserialized, TestCaseEditorDataFrameSchema, trusted=True)
        return cast(TestCaseEditorDataFrame, df_validated)


class MetricsDataFrameSchema(pa.DataFrameModel):
    test_sample: Optional[Series[JSONObject]] = pa.Field(coerce=True)
    test_case_id: Optional[Series[pa.typing.Int64]] = pa.Field(coerce=True)
    configuration_display_name: Optional[Series[pa.typing.String]] = pa.Field(coerce=True, nullable=True)
    metrics: Series[JSONObject] = pa.Field(coerce=True)


class MetricsDataFrame(LoadableDataFrame[MetricsDataFrameSchema]):
    @classmethod
    def get_schema(cls) -> Type[MetricsDataFrameSchema]:
        return MetricsDataFrameSchema

    def as_serializable(self) -> pd.DataFrame:
        return MetricsDataFrame._serde(self, True)

    @classmethod
    def from_serializable(cls, df: pd.DataFrame) -> "MetricsDataFrame":
        df_deserialized = MetricsDataFrame._serde(df, False)
        df_validated = validate_df_schema(df_deserialized, MetricsDataFrameSchema, trusted=True)
        return cast(MetricsDataFrame, df_validated)

    @staticmethod
    def _serde(df: pd.DataFrame, serialize: bool) -> pd.DataFrame:
        serde_function = as_serialized_json if serialize else as_deserialized_json
        df_out = df.copy()
        if "test_sample" in df.columns:
            df_out["test_sample"] = df_out["test_sample"].apply(serde_function)
        df_out["metrics"] = df_out["metrics"].apply(serde_function)
        return df_out
