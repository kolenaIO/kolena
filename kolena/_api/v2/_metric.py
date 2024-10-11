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
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

from pydantic import Field
from pydantic import field_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Annotated

MetricFormat = Literal["default", "integer", "decimal", "percentage", "scientific", "dollars", "euros"]


@dataclass(frozen=True, order=True)
class Highlight:
    higherIsBetter: Union[bool, None] = None


@dataclass(frozen=True)
class BaseMetric:
    label: str
    source: Literal["result"]


AveragingMethod = Literal["macro", "micro", "weighted"]
ClassifierAggregator = Literal["precision", "recall", "f1_score"]
SimpleAggregator = Literal["count", "min", "max", "mean", "median", "stddev", "sum"]
Source = Literal["datapoint", "result"]


@dataclass(frozen=True)
class SimpleParams:
    key: str


@dataclass(frozen=True)
class SimpleMetric(BaseMetric):
    aggregator: SimpleAggregator
    params: SimpleParams
    format: Union[MetricFormat, None] = None
    highlight: Union[Highlight, None] = None


@dataclass(frozen=True)
class CategoricalAggregator:
    aggregator: Literal["count", "rate"]
    fieldValue: Union[bool, str]


@dataclass(frozen=True)
class CategoricalMetric(BaseMetric):
    aggregator: CategoricalAggregator
    params: SimpleParams
    format: Union[MetricFormat, None] = None
    highlight: Union[Highlight, None] = None


@dataclass(frozen=True, order=True)
class BinaryClassificationAggregator:
    aggregator: Union[Literal["accuracy"], Literal["auc"], ClassifierAggregator]
    type: Literal["binary-classification"]


@dataclass(frozen=True, order=True)
class MulticlassClassificationAggregator:
    aggregator: Union[Literal["accuracy"], ClassifierAggregator]
    averagingMethod: Annotated[Union[AveragingMethod, None], Field(validate_default=True)] = None

    @field_validator("averagingMethod")
    @classmethod
    def set_null_averaging_method(cls, v: Union[str, None]) -> str:
        return v or "macro"


@dataclass(frozen=True, order=True)
class RegressionAggregator:
    aggregator: Literal["mae", "mse", "rmse", "r_squared", "pearson_corr", "spearman_corr"]


@dataclass(frozen=True, order=True)
class ClassificationParams:
    groundTruthField: str
    inferenceField: str
    threshold: Union[int, float, None] = None
    include_auc: bool = False


@dataclass(frozen=True)
class ClassificationMetric(BaseMetric):
    aggregator: Union[BinaryClassificationAggregator, MulticlassClassificationAggregator, RegressionAggregator]
    params: ClassificationParams
    format: Union[MetricFormat, None] = None
    highlight: Union[Highlight, None] = None


@dataclass(frozen=True, order=True)
class ObjectDetectionAggregator:
    aggregator: Union[Literal["average_precision"], ClassifierAggregator]
    averagingMethod: Annotated[Union[AveragingMethod, None], Field(validate_default=True)] = None

    @field_validator("averagingMethod")
    @classmethod
    def set_null_averaging_method(cls, v: Union[str, None]) -> str:
        return v or "macro"


@dataclass(frozen=True, order=True)
class ObjectDetectionParams:
    fnField: str
    fpField: str
    tpField: str
    scoreField: str
    labelField: Union[str, None] = None


@dataclass(frozen=True)
class ObjectDetectionMetric(BaseMetric):
    aggregator: ObjectDetectionAggregator
    params: ObjectDetectionParams
    format: Union[MetricFormat, None] = None
    highlight: Union[Highlight, None] = None


@dataclass(frozen=True, order=True)
class ThresholdedObjectPoweredAggregator:
    aggregator: Union[Literal["average_precision"], ClassifierAggregator]
    averagingMethod: AveragingMethod = "macro"
    type: Literal["thresholded-object-powered"] = "thresholded-object-powered"


@dataclass(frozen=True, order=True)
class ThresholdedObjectPoweredParams:
    threshold: float
    thresholded_object_field: str
    precision_tp_field: str
    precision_fp_field: str
    recall_tp_field: str
    recall_fn_field: str
    label: Optional[str] = None


@dataclass(frozen=True)
class ThresholdedObjectPoweredMetric(BaseMetric):
    aggregator: ThresholdedObjectPoweredAggregator
    params: ThresholdedObjectPoweredParams
    format: Union[MetricFormat, None] = None
    highlight: Union[Highlight, None] = None


@dataclass(frozen=True)
class PrecisionParams:
    fp: str
    tp: str


@dataclass(frozen=True)
class PrecisionMetric(BaseMetric):
    aggregator: Literal["precision"]
    params: PrecisionParams
    format: Union[MetricFormat, None] = None
    highlight: Union[Highlight, None] = None


@dataclass(frozen=True)
class RecallParams:
    fn: str
    tp: str


@dataclass(frozen=True)
class RecallMetric(BaseMetric):
    aggregator: Literal["recall"]
    params: RecallParams
    format: Union[MetricFormat, None] = None
    highlight: Union[Highlight, None] = None


@dataclass(frozen=True)
class F1Params:
    fn: str
    fp: str
    tp: str


@dataclass(frozen=True)
class F1Metric(BaseMetric):
    aggregator: Literal["f1_score"]
    params: F1Params
    format: Union[MetricFormat, None] = None
    highlight: Union[Highlight, None] = None


@dataclass(frozen=True)
class AccuracyParams:
    fp: str
    fn: str
    tp: str
    tn: str


@dataclass(frozen=True)
class AccuracyMetric(BaseMetric):
    aggregator: Literal["accuracy"]
    params: AccuracyParams
    format: Union[MetricFormat, None] = None
    highlight: Union[Highlight, None] = None


@dataclass(frozen=True)
class RateParams:
    fp: str
    tn: str


@dataclass(frozen=True)
class RateMetric(BaseMetric):
    aggregator: Literal["false_positive_rate", "true_negative_rate"]
    params: RateParams
    format: Union[MetricFormat, None] = None
    highlight: Union[Highlight, None] = None


@dataclass(frozen=True)
class WeightedMetricAggregator:
    type: Literal["weighted-metric"] = "weighted-metric"
    aggregator: Literal["mean"] = "mean"


@dataclass(frozen=True)
class WeightedMetricParams:
    metric_field: str
    weight_field: str
    # TODO: mark these as mandatory when marina supports non-result sources
    metric_source: Source = "result"
    weight_source: Source = "result"


@dataclass(frozen=True)
class WeightedMetric(BaseMetric):
    aggregator: WeightedMetricAggregator
    params: WeightedMetricParams
    highlight: Union[Highlight, None] = None


FormulaMetric = Union[
    SimpleMetric,
    CategoricalMetric,
    PrecisionMetric,
    RecallMetric,
    F1Metric,
    AccuracyMetric,
    RateMetric,
    WeightedMetric,
]

TaskMetric = Union[ClassificationMetric, ObjectDetectionMetric, ThresholdedObjectPoweredMetric]

Metric = Union[
    FormulaMetric,
    TaskMetric,
]


@dataclass(frozen=True)
class MetricGroup:
    name: str
    metrics: List[Metric]

    @field_validator("metrics")
    @classmethod
    def metric_label_unique(cls, metrics: List[Metric]) -> List[Metric]:
        if len(metrics) > len({metric.label for metric in metrics}):
            raise ValueError("Metric labels must be unique.")
        return metrics

    def __hash__(self) -> int:
        return hash(
            (
                self.name,
                tuple(self.metrics),
            ),
        )


@dataclass(frozen=True)
class DatasetMetricGroups:
    dataset_id: int
    dataset_name: str
    metric_groups: List[MetricGroup]
