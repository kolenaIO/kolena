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
from abc import ABCMeta
from dataclasses import dataclass
from dataclasses import fields

from kolena.workflow._datatypes import TypedDataObject


class PreventThresholdOverrideMeta(ABCMeta, type):
    def __new__(cls, name, bases, dct):
        if "threshold" in dct.get("__annotations__", {}):
            for base in bases:
                if base.__name__ == "ThresholdedMetrics":
                    raise TypeError(f"Subclasses of {base.__name__} cannot override 'threshold'")
        return super().__new__(cls, name, bases, dct)


@dataclass(frozen=True)
class ThresholdedMetrics(TypedDataObject, metaclass=PreventThresholdOverrideMeta):
    """
    A data class representing metrics that have a specified threshold.

    This class does not allow dictionary objects as field values and ensures
    that the threshold cannot be overridden.

    Attributes:
        threshold (float): The threshold value for the metrics.

    Usage:
        # Create an instance of ThresholdedMetrics
        metrics = ThresholdedMetrics(threshold=0.5)

    Raises:
        TypeError: If any field value is a dictionary.
    """

    threshold: float

    def _data_type() -> str:
        return "METRICS/THRESHOLDED"

    def __post_init__(self) -> None:
        for field in fields(self):
            field_value = getattr(self, field.name)
            if isinstance(field_value, dict):
                raise TypeError(f"Field '{field.name}' should not be a dictionary")
