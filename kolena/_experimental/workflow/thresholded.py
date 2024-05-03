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
from abc import ABCMeta
from dataclasses import dataclass
from dataclasses import fields

from kolena._utils.datatypes import DataCategory
from kolena._utils.datatypes import DataType
from kolena._utils.datatypes import TypedDataObject


class PreventThresholdOverrideMeta(ABCMeta, type):
    def __new__(cls, name: str, bases: tuple, dct: dict) -> "PreventThresholdOverrideMeta":
        if "threshold" in dct.get("__annotations__", {}):
            for base in bases:
                if base.__name__ == "ThresholdedMetrics":
                    raise TypeError(f"Subclasses of {base.__name__} cannot override 'threshold'")
        return super().__new__(cls, name, bases, dct)


class _MetricsType(DataType):
    THRESHOLDED = "THRESHOLDED"

    @staticmethod
    def _data_category() -> DataCategory:
        return DataCategory.METRICS


@dataclass(frozen=True)
class ThresholdedMetrics(TypedDataObject[_MetricsType], metaclass=PreventThresholdOverrideMeta):
    """
    Represents metrics tied to a specific threshold.

    `List[ThresholdedMetrics]` should be used as a field type within `MetricsTestSample`
    from the `kolena.workflow` module. This list is meant to hold metric values
    associated with distinct thresholds. These metrics are expected to be uniform across `TestSample`
    instances within a single test execution.

    `ThresholdedMetrics` prohibits the use of dictionary objects as field values and guarantees that
    the threshold values remain immutable once set. For application within a particular workflow,
    subclassing is required to define relevant metrics fields.

    Usage example:

    ```python
    from kolena.workflow import MetricsTestSample
    from kolena._experimental.workflow import ThresholdedMetrics

    @dataclass(frozen=True)
    class ClassThresholdedMetrics(ThresholdedMetrics):
        precision: float
        recall: float
        f1: float

    @dataclass(frozen=True)
    class TestSampleMetrics(MetricsTestSample):
        car: List[ClassThresholdedMetrics]
        pedestrian: List[ClassThresholdedMetrics]

    # Creating an instance of metrics
    metric = TestSampleMetrics(
        car=[
            ClassThresholdedMetrics(threshold=0.3, precision=0.5, recall=0.8, f1=0.615),
            ClassThresholdedMetrics(threshold=0.4, precision=0.6, recall=0.6, f1=0.6),
            ClassThresholdedMetrics(threshold=0.5, precision=0.8, recall=0.4, f1=0.533),
            # ...
        ],
        pedestrian=[
            ClassThresholdedMetrics(threshold=0.3, precision=0.6, recall=0.9, f1=0.72),
            ClassThresholdedMetrics(threshold=0.4, precision=0.7, recall=0.7, f1=0.7),
            ClassThresholdedMetrics(threshold=0.5, precision=0.8, recall=0.6, f1=0.686),
            # ...
        ],
    )
    ```

    Raises:
        TypeError: If any of the field values is a dictionary.
    """

    threshold: float

    @classmethod
    def _data_type(cls) -> _MetricsType:
        """
        Returns the type of metrics represented by this class.

        Returns:
            _MetricsType: An enumeration value indicating the type of metrics.
        """
        return _MetricsType.THRESHOLDED

    def __post_init__(self) -> None:
        for field in fields(self):
            field_value = getattr(self, field.name)
            if isinstance(field_value, dict):
                raise TypeError(f"Field '{field.name}' should not be a dictionary")
