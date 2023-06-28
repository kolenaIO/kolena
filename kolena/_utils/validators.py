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
from pydantic import Extra


class ValidatorConfig:
    """Pydantic configuration for dataclasses and @validate_arguments decorators."""

    arbitrary_types_allowed = True
    smart_union = True
    extra = Extra.allow  # do not fail when unrecognized values are provided


def validate_not_blank(field: str):
    if not bool(field and not field.isspace()):
        raise ValueError("Names must be non empty")
