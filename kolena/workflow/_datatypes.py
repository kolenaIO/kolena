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
import dataclasses
from abc import ABCMeta
from abc import abstractmethod
from collections import OrderedDict
from enum import Enum
from typing import Any
from typing import cast
from typing import Dict
from typing import Generic
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import Series
from pydantic.dataclasses import dataclass

from kolena._utils.dataframes.validators import validate_df_schema
from kolena._utils.datatypes import get_args
from kolena._utils.datatypes import get_origin
from kolena._utils.datatypes import LoadableDataFrame
from kolena._utils.serde import as_deserialized_json
from kolena._utils.serde import as_serialized_json
from kolena._utils.serde import with_serialized_columns
from kolena._utils.validators import ValidatorConfig

T = TypeVar("T", bound="DataObject")


@dataclass(frozen=True, config=ValidatorConfig)
class DataObject(metaclass=ABCMeta):
    def _to_dict(self) -> Dict[str, Any]:
        def serialize_value(value: Any) -> Any:
            if isinstance(value, (bool, str, int, float)) or value is None:
                return value
            if isinstance(value, np.generic):  # numpy scalars are common enough to be worth specific handling
                for base_type, numpy_type in [(bool, np.bool_), (int, np.integer), (float, np.inexact)]:
                    if isinstance(value, numpy_type):  # cast if there is a match, otherwise fallthrough
                        return base_type(value)
            if isinstance(value, DataObject):
                return value._to_dict()
            if isinstance(value, (list, tuple, np.ndarray)):
                return [serialize_value(subvalue) for subvalue in value]
            if isinstance(value, dict):
                return {key: serialize_value(subvalue) for key, subvalue in value.items()}
            raise ValueError(f"unsupported value type: '{type(value).__name__}' (value: {value})")

        items = [(field.name, getattr(self, field.name)) for field in dataclasses.fields(type(self))]
        return OrderedDict([(key, serialize_value(value)) for key, value in items])

    @classmethod
    def _from_dict(cls: Type[T], obj_dict: Dict[str, Any]) -> T:
        def deserialization_invalid_value(field_name: str, field_type: Type, field_value: Any) -> ValueError:
            return ValueError(f"invalid value '{field_value}' provided for field '{field_name}' of type '{field_type}'")

        def default_value_or_raise(field: dataclasses.Field, field_value: Any) -> Any:
            if field.default is not dataclasses.MISSING:
                return field.default
            if field.default_factory is not dataclasses.MISSING:
                return field.default_factory()
            raise deserialization_invalid_value(field.name, field.type, field_value)

        def deserialize_field(
            field: dataclasses.Field,
            field_value: Any,
            field_type: Optional[Type] = None,  # override field.type, for recursively deserializing e.g. List[T]
            attempt_cast: bool = True,
        ) -> Any:
            """
            Custom deserialization following rules applied in ``validate_data_object_type``. This is maintained as a
            faster alternative to deserialize an arbitrarily nested value into a dataclass.
            """
            field_type = field_type or field.type
            origin = get_origin(field_type)  # non-None for typing.X types

            if origin is list:
                (arg,) = get_args(field_type)
                if not isinstance(field_value, list):
                    return default_value_or_raise(field, field_value)
                return [deserialize_field(field, v, field_type=arg) for v in field_value]

            elif origin is dict or origin is Dict:
                (arg_key, arg_val) = get_args(field_type)
                if not isinstance(field_value, dict):
                    return default_value_or_raise(field, field_value)
                entries = [
                    (deserialize_field(field, k, field_type=arg_key), deserialize_field(field, v, field_type=arg_val))
                    for k, v in field_value.items()
                ]
                return OrderedDict(entries)

            elif origin is Union:  # includes Optional[T]
                args = get_args(field_type)
                # first attempt to find a direct match without casting, then if that fails try each type with casting
                for should_cast in [False, True]:
                    for arg in args:  # find the first type arg in the list for which deserialization succeeds
                        try:
                            return deserialize_field(field, field_value, field_type=arg, attempt_cast=should_cast)
                        except (ValueError, AttributeError):  # potential AttributeError when field_value=None
                            continue
                raise deserialization_invalid_value(field.name, field_type, field_value)

            elif origin is tuple or origin is Tuple:
                args = get_args(field_type)
                if not isinstance(field_value, (list, tuple)):  # should really only be list as the input is JSON
                    raise deserialization_invalid_value(field.name, field_type, field_value)
                return tuple(deserialize_field(field, field_value[i], field_type=arg) for i, arg in enumerate(args))

            elif origin is not None:  # unsupported generic
                raise ValueError(f"unsupported type '{field_type}' for field '{field.name}' and value '{field_value}'")

            # if we've reached here, it's not a typing generic, and issubclass is safe to use
            if field_value is None and not issubclass(field_type, type(None)):
                return default_value_or_raise(field, field_value)

            if isinstance(field_value, field_type):
                return field_value

            elif issubclass(field_type, DataObject):
                return field_type._from_dict(field_value)

            if attempt_cast and not issubclass(field_type, type(None)):
                return field_type(field_value)

            raise ValueError(
                f"cast=False, not casting field '{field.name}' to type '{field_type}' with value '{field_value}'",
            )

        items = {f.name: deserialize_field(f, obj_dict.get(f.name, None)) for f in dataclasses.fields(cls)}
        return cls(**items)


class DataType(str, Enum):
    @staticmethod
    def _data_category() -> str:
        raise NotImplementedError


DATA_TYPE_FIELD = "data_type"
U = TypeVar("U", bound=DataType)


@dataclass(frozen=True, config=ValidatorConfig)
class TypedDataObject(Generic[U], DataObject, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def _data_type() -> U:
        raise NotImplementedError

    def _to_dict(self) -> Dict[str, Any]:
        self_dict = super()._to_dict()
        self_type = self._data_type()
        self_dict[DATA_TYPE_FIELD] = f"{self_type._data_category()}/{self_type.value}"
        return self_dict


_SCALAR_TYPES = [str, bool, int, float]
JSONObject = object


class TestSampleDataFrameSchema(pa.SchemaModel):
    """General-purpose frame used for test samples in isolation or paired with ground truths and/or inferences."""

    test_sample: Series[JSONObject] = pa.Field(coerce=True)
    test_sample_metadata: Optional[Series[JSONObject]] = pa.Field(coerce=True)
    ground_truth: Optional[Series[JSONObject]] = pa.Field(coerce=True, nullable=True)
    inference: Optional[Series[JSONObject]] = pa.Field(coerce=True)


class TestSampleDataFrame(
    pa.typing.DataFrame[TestSampleDataFrameSchema],
    LoadableDataFrame["TestSampleDataFrame"],
):
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


class TestSuiteTestSamplesDataFrameSchema(pa.SchemaModel):
    """Data frame used for loading test samples grouped by test case."""

    test_case_id: Optional[Series[pa.typing.Int64]] = pa.Field(coerce=True)
    test_sample: Series[JSONObject] = pa.Field(coerce=True)


class TestSuiteTestSamplesDataFrame(
    pa.typing.DataFrame[TestSuiteTestSamplesDataFrameSchema],
    LoadableDataFrame["TestSuiteTestSamplesDataFrame"],
):
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
        return df_out


class TestCaseEditorDataFrameSchema(pa.SchemaModel):
    test_sample_type: Series[pa.typing.String] = pa.Field(coerce=True)
    test_sample: Series[JSONObject] = pa.Field(coerce=True)  # TODO: validators?
    test_sample_metadata: Series[JSONObject] = pa.Field(coerce=True)
    ground_truth: Series[JSONObject] = pa.Field(coerce=True, nullable=True)
    remove: Series[pa.typing.Bool] = pa.Field(coerce=True)


class TestCaseEditorDataFrame(
    pa.typing.DataFrame[TestCaseEditorDataFrameSchema],
    LoadableDataFrame["TestCaseEditorDataFrame"],
):
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


class MetricsDataFrameSchema(pa.SchemaModel):
    test_sample: Optional[Series[JSONObject]] = pa.Field(coerce=True)
    test_case_id: Optional[Series[pa.typing.Int64]] = pa.Field(coerce=True)
    configuration_display_name: Optional[Series[pa.typing.String]] = pa.Field(coerce=True, nullable=True)
    metrics: Series[JSONObject] = pa.Field(coerce=True)


class MetricsDataFrame(pa.typing.DataFrame[MetricsDataFrameSchema], LoadableDataFrame["MetricsDataFrame"]):
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
