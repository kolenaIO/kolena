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
from kolena._utils.pydantic_v1 import validate_arguments
from kolena._utils.validators import ValidatorConfig


@validate_arguments(config=ValidatorConfig)
def validate_label(label: str) -> None:
    if label.strip() == "":
        raise ValueError("label must contain non-whitespace characters", label)


@validate_arguments(config=ValidatorConfig)
def validate_confidence(confidence: float) -> None:
    if not (0 <= confidence <= 1):
        raise ValueError("confidence must be between 0 and 1 (inclusive)", confidence)
