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
import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

from kolena._utils.datatypes import get_args
from kolena._utils.datatypes import get_origin
from kolena.errors import WorkflowMismatchError
from kolena.workflow._datatypes import _SCALAR_TYPES
from kolena.workflow._datatypes import DATA_TYPE_FIELD
from kolena.workflow._datatypes import DataObject
from kolena.workflow.annotation import _ANNOTATION_TYPES
from kolena.workflow.asset import _ASSET_TYPES

_SUPPORTED_FIELD_TYPES = [*_SCALAR_TYPES, *_ANNOTATION_TYPES, *_ASSET_TYPES]


def assert_workflows_match(workflow_expected: str, workflow_provided: str) -> None:
    if workflow_provided != workflow_expected:
        raise WorkflowMismatchError(
            f"workflow '{workflow_provided}' does not match expected workflow '{workflow_expected}'",
        )


def validate_data_object_type(data_object_type: Type[DataObject]) -> None:
    if not issubclass(data_object_type, DataObject):
        raise ValueError(f"'{data_object_type.__name__}' must extend {DataObject.__name__}")

    # note that we use obj.__annotations__ instead of dataclasses.fields(obj) as the latter is not yet populated by the
    # time __init_subclass__ is called, blocking usage of this validator in __init_subclass__
    fields = getattr(data_object_type, "__annotations__", None)
    if fields is None:
        fields = {field.name: field.type for field in dataclasses.fields(data_object_type)}
    for field_name, field_type in fields.items():
        validate_field(field_name, field_type)


def validate_scalar_data_object_type(data_object_type: Type[DataObject]) -> None:
    if not issubclass(data_object_type, DataObject):
        raise ValueError(f"'{data_object_type.__name__}' must extend {DataObject.__name__}")

    fields = getattr(data_object_type, "__annotations__", None)
    if fields is None:
        fields = {field.name: field.type for field in dataclasses.fields(data_object_type)}
    for field_name, field_type in fields.items():
        validate_field(field_name, field_type, supported_field_types=_SCALAR_TYPES, allow_lists=False)


def validate_field(
    field_name: str,
    field_type: Type,
    supported_field_types: Optional[List[Type]] = None,
    allow_lists: bool = True,
) -> None:
    if field_name == DATA_TYPE_FIELD:
        raise ValueError(f"Unsupported field name: '{DATA_TYPE_FIELD}' is reserved")

    supported_field_types = supported_field_types or _SUPPORTED_FIELD_TYPES
    supported_bases = ", ".join(t.__name__ for t in supported_field_types)

    origin = get_origin(field_type)
    if origin is list:
        if not allow_lists:
            raise ValueError(f"Unsupported type for field: field '{field_name}', type '{field_type}'")
        validate_list(field_name, field_type, supported_field_types)

    elif origin is Union:
        validate_union(field_name, field_type, supported_field_types, allow_lists)

    elif origin is not None or field_type == typing.Any:
        raise ValueError(f"Unsupported field type: field '{field_name}', type '{field_type}'")

    elif not issubclass(field_type, tuple(supported_field_types)):
        raise ValueError(
            f"Unsupported field type: field '{field_name}', type '{field_type.__name__}'\n"
            f"Supported base types for fields: {supported_bases}",
        )


def validate_list(field_name: str, field_type: List, supported_field_types: List[Type]) -> None:
    (arg,) = get_args(field_type)
    arg_origin = get_origin(arg)
    if arg_origin is Union:
        validate_union(field_name, arg, supported_field_types, True)
    elif arg_origin is not None or not issubclass(arg, tuple(supported_field_types)):
        raise ValueError(
            f"Unsupported field type: field '{field_name}', unsupported List type '{arg}'",
        )


def validate_union(field_name: str, field_type: Union, supported_field_types: List[Type], allow_lists: bool) -> None:
    args = get_args(field_type)
    type_name = "Optional" if type(None) in args and len(args) == 2 else "Union"
    for arg in args:
        err = f"Unsupported field type: field '{field_name}', unsupported {type_name} type '{arg}'"
        arg_origin = get_origin(arg)
        if arg_origin is list or arg_origin is List:
            if not allow_lists:
                raise ValueError(err)
            validate_list(field_name, arg, supported_field_types)
            continue
        if arg_origin is not None or arg == typing.Any:  # e.g. Optional, Dict
            raise ValueError(err)
        if issubclass(arg, type(None)):
            continue
        if not issubclass(arg, tuple(supported_field_types)):
            raise ValueError(err)


def validate_metadata_dict(field_type: Type) -> None:
    origin = get_origin(field_type)
    if not (origin is dict or origin is Dict):
        raise ValueError("Metadata field must be dictionary type")
    args = get_args(field_type)
    if len(args) < 2:
        raise ValueError("Metadata dictionary must specify key and value types")
    if args[0] != str:
        raise ValueError(f"metadata.key type must be str: '{args[0]}'")
    validate_field("metadata.value", args[1], supported_field_types=[*_SCALAR_TYPES, type(None)])


def safe_issubclass(cls: Any, clsinfo: Any) -> bool:
    try:
        return issubclass(cls, clsinfo)
    except Exception:
        return False
