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
import enum
from dataclasses import field
from typing import Dict
from typing import List
from typing import Optional

from pydantic import conlist
from pydantic import model_validator
from pydantic.dataclasses import dataclass

from kolena._api.v2._api import GeneralFieldFilter
from kolena._api.v2._api import Range
from kolena._api.v2._derived_field import DerivedField
from kolena._api.v2._dsl import Dsl
from kolena._api.v2._metric import MetricGroup
from kolena.errors import IncorrectUsageError


@dataclass(frozen=True)
class ModelFilter:
    id: int
    eval_config_id: int
    result: Optional[Dict[str, GeneralFieldFilter]] = None
    is_null: Optional[bool] = None

    def __hash__(self) -> int:
        return hash(
            (
                self.id,
                self.eval_config_id,
                tuple(self.result or []),
                self.is_null,
            ),
        )

    @model_validator(mode="after")
    def validate_stratify_field_or_filter(self) -> "ModelFilter":
        if self.result and self.is_null:
            raise ValueError("Conflicting value filter and null filter.")
        return self


@dataclass(frozen=True)
class HumanEvaluationFilter:
    id: str
    human_evaluation: Optional[Dict[str, GeneralFieldFilter]] = None

    def __hash__(self) -> int:
        return hash(
            (
                self.id,
                tuple(self.human_evaluation or []),
            ),
        )


@dataclass(frozen=True)
class DifficultyScoreFilter:
    metric_groups: List[MetricGroup]

    def __hash__(self) -> int:
        return hash(
            (tuple(self.metric_groups),),
        )


class RelationshipOperator(str, enum.Enum):
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    GREATER_THAN = "greater_than"
    SMALLER_THAN = "smaller_than"
    DIFFERENCE_RANGE = "difference_range"


class ComparisonObjectType(str, enum.Enum):
    RESULT = "result"
    DATAPOINT = "datapoint"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ComparisonFilterDataSource:
    field: str
    type: ComparisonObjectType
    model_id: Optional[int] = None
    eval_config_id: Optional[int] = None


@dataclass(frozen=True)
class Relationship:
    operator: RelationshipOperator
    difference_range: Optional[Range] = None


@dataclass(frozen=True)
class CompareFilter:
    left: ComparisonFilterDataSource
    right: ComparisonFilterDataSource
    relationship: Relationship


@dataclass(frozen=True)
class Filters:
    dataset_ids: conlist(int, min_length=1)
    datapoint_ids: Optional[conlist(int, min_length=0)] = None
    datapoint: Dict[str, GeneralFieldFilter] = field(default_factory=dict)
    models: List[ModelFilter] = field(default_factory=list)
    compare_filters: List[CompareFilter] = field(default_factory=list)
    human_evaluations: conlist(HumanEvaluationFilter, max_length=1) = field(default_factory=list)
    difficulty_scores: Optional[DifficultyScoreFilter] = None
    dsl: Optional[Dsl] = None
    derived_fields: Optional[List[DerivedField]] = None

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.dataset_ids),
                tuple(self.datapoint),
                tuple(self.models),
                tuple(self.compare_filters),
                tuple(self.human_evaluations),
                self.difficulty_scores,
                self.dsl,
                tuple(self.derived_fields or []),
            ),
        )

    def __post_init__(self) -> None:
        all_evaluations = [(model.id, model.eval_config_id) for model in self.models]
        unique_evaluations = set(all_evaluations)
        if len(all_evaluations) != len(unique_evaluations):
            raise IncorrectUsageError("duplicate model and eval_config_id")
        if self.datapoint_ids:
            if len(set(self.datapoint_ids)) != len(self.datapoint_ids):
                raise IncorrectUsageError("duplicate datapoint ids")
        if self.difficulty_scores and not self.models:
            raise IncorrectUsageError("difficulty_scores cannot be used when models are not provided")
