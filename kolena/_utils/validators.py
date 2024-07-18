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
from typing import Optional

from kolena._utils.consts import FieldName
from kolena._utils.pydantic_v1 import Extra


# Pydantic configuration for dataclasses and @validate_arguments decorators
class ValidatorConfig:
    arbitrary_types_allowed = True
    smart_union = True
    extra = Extra.allow  # do not fail when unrecognized values are provided


def validate_name(field: str, field_name: Optional[FieldName] = None) -> None:
    field_name_str = field_name.value if field_name else "field"
    if not field or field.isspace():
        raise ValueError(f"{field_name_str} must be non empty")
