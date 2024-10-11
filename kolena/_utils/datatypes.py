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
import importlib
from abc import ABC
from abc import ABCMeta
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from enum import Enum
from typing import Any
from typing import cast
from typing import Dict
from typing import Generic
from typing import get_args
from typing import get_origin
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

import numpy as np
import pandas as pd
import pandera as pa

from kolena._utils import log
from kolena._utils.dataframes.validators import validate_df_schema
from kolena._utils.pydantic_v1 import Extra
from kolena._utils.pydantic_v1.dataclasses import dataclass
from kolena._utils.validators import ValidatorConfig


TDataFrame = TypeVar("TDataFrame", bound="LoadableDataFrame")
TSchema = TypeVar("TSchema", bound=pa.DataFrameModel)


class LoadableDataFrame(ABC, pa.typing.DataFrame[TSchema], Generic[TSchema]):
    @classmethod
    @abstractmethod
    def get_schema(cls) -> Type[TSchema]:
        raise NotImplementedError

    @classmethod
    def construct_empty(cls: Type[TDataFrame]) -> TDataFrame:
        df = pd.DataFrame({key: [] for key in cls.get_schema().to_schema().columns.keys()})
        return cast(TDataFrame, validate_df_schema(df, cls.get_schema(), trusted=True))

    @classmethod
    def from_serializable(cls: Type[TDataFrame], df: pd.DataFrame) -> TDataFrame:
        return cast(TDataFrame, validate_df_schema(df, cls.get_schema(), trusted=True))


T = TypeVar("T", bound="DataObject")


def _double_under(input: str) -> bool:
    return input.startswith("__") and input.endswith("__")


def _allow_extra(cls: Type[T]) -> bool:
    # `pydantic.dataclasses.is_built_in_dataclass` would have false-positive when a stdlib-dataclass decorated
    # class extends a pydantic dataclass
    return "__pydantic_model__" in vars(cls) and cls.__pydantic_model__.Config.extra == Extra.allow  # type: ignore


# used to track data_type string -> TypedDataObject
_DATA_TYPE_MAP: Dict[str, Type["TypedDataObject"]] = {}


class DataCategory(str, Enum):
    TEST_SAMPLE = "TEST_SAMPLE"
    PLOT = "PLOT"
    METRICS = "METRICS"
    ASSET = "ASSET"
    ANNOTATION = "ANNOTATION"

    def data_category_to_module_name(self) -> str:
        if self == DataCategory.TEST_SAMPLE:
            return "kolena.workflow"
        if self == DataCategory.PLOT:
            return "kolena.workflow.plot"
        if self == DataCategory.METRICS:
            return "kolena._experimental.workflow.thresholded"
        if self == DataCategory.ASSET:
            return "kolena.asset"
        if self == DataCategory.ANNOTATION:
            return "kolena.annotation"
        raise ValueError(f"Must specify module name for data category: {self}")


def _get_full_type(obj: Type["TypedDataObject"]) -> str:
    data_type = obj._data_type()
    return f"{data_type._data_category().value}/{data_type.value}"


def _get_data_type(name: str) -> Optional[Type["TypedDataObject"]]:
    class_type = _DATA_TYPE_MAP.get(name, None)
    if not class_type:
        try:
            data_category, _ = name.split("/")
            module = DataCategory(data_category).data_category_to_module_name()
            importlib.import_module(module)
        except Exception as e:
            log.error(f"Failed to import module for data type '{name}'", e)
        finally:
            return _DATA_TYPE_MAP.get(name, None)
    return class_type


# used for TypedBaseDataObject to register themselves to be used in dataclass extra fields deserialization
def _register_data_type(cls: Type["TypedDataObject"]) -> None:
    full_name = _get_full_type(cls)
    # leverage class inheritance order, only keep base classes of a datatype
    if full_name not in _DATA_TYPE_MAP:
        _DATA_TYPE_MAP[full_name] = cls


def _deserialize_typed_dataobject(value: Dict[Any, Any]) -> Any:
    data_type = _get_data_type(value[DATA_TYPE_FIELD])
    if data_type is None:
        return value

    # 'data_type' is not a real member
    value.pop(DATA_TYPE_FIELD)
    return data_type._from_dict(value)


# best effort to deserialize typed data objects in dataclass extra fields
# note: since a "data_type" string could map to multiple classes through inheritance, only base case would be used.
def _try_deserialize_typed_dataobject(value: Any) -> Any:
    if isinstance(value, list):
        # only attempt deserialization when it is likely this is a list of typed data objects
        if value and isinstance(value[0], dict) and DATA_TYPE_FIELD in value[0]:
            return [_try_deserialize_typed_dataobject(val) for val in value]
    elif isinstance(value, dict):
        if DATA_TYPE_FIELD in value:
            return _deserialize_typed_dataobject(value)

    return value


@dataclass(frozen=True, config=ValidatorConfig)
class DataObject(metaclass=ABCMeta):
    """The base for various objects in `kolena.workflow`."""

    def __str__(self) -> str:
        if not _allow_extra(type(self)):
            return self.__repr__()

        # emulate stdlib dataclass _repr_fn implementation, extending extra fields
        fields = [f.name for f in dataclasses.fields(self)]
        extras = [f for f in vars(self) if f not in fields and not _double_under(f)]

        # caveat: dataclass generate `__repr__` for decorated classes, as such calls `__str__` for `DataObject` instead.
        value_str = ", ".join(
            [
                f"{f}={getattr(self, f)}" if isinstance(getattr(self, f), DataObject) else f"{f}={getattr(self, f)!r}"
                for f in fields + extras
            ],
        )
        return f"{self.__class__.__qualname__}({value_str})"

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
        field_names = {field.name for field in dataclasses.fields(type(self))}
        if _allow_extra(type(self)):
            for key, val in vars(self).items():
                if key not in field_names and not _double_under(key):
                    items.append((key, val))
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

            if origin is list or origin is Sequence:
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

        items = {f.name: deserialize_field(f, obj_dict.get(f.name, None)) for f in dataclasses.fields(cls) if f.init}
        field_names = {f.name for f in dataclasses.fields(cls)}
        if _allow_extra(cls):
            for key, val in obj_dict.items():
                if key not in field_names:
                    items[key] = _try_deserialize_typed_dataobject(val)
        return cls(**items)  # type: ignore

    # integrate with pandas json deserialization
    # https://pandas.pydata.org/docs/user_guide/io.html#fallback-behavior
    def toDict(self) -> Dict:
        return self._to_dict()


def _serialize_dataobject(x: Any) -> Any:
    if isinstance(x, list):
        return [item._to_dict() if isinstance(item, DataObject) else item for item in x]

    return x._to_dict() if isinstance(x, DataObject) else x


class DataType(str, Enum):
    @staticmethod
    def _data_category() -> DataCategory:
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
        self_dict[DATA_TYPE_FIELD] = f"{self_type._data_category().value}/{self_type.value}"
        return self_dict

    def __init_subclass__(cls, **kwargs: Any):
        try:
            _register_data_type(cls)
        except NotImplementedError:
            pass
