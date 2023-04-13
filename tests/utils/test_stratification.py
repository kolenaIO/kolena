from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import List
from typing import Type

import pandas as pd
import pytest
from pydantic import ValidationError

from kolena._api.v1.core import BaseStratifyRequest
from kolena._api.v1.core import CATEGORICAL
from kolena._api.v1.core import CategoricalBucket
from kolena._api.v1.core import Dimension
from kolena._api.v1.core import DimensionSpec
from kolena._api.v1.core import IntervalType
from kolena._api.v1.core import NUMERIC
from kolena._api.v1.core import NumericBucket
from kolena._utils.stratification import stratify

DIM_DUMMY_NUMERIC = Dimension("test_sample", "a", NUMERIC)
DIM_DUMMY_CATEGORICAL = Dimension("test_sample", "a", CATEGORICAL)

DF_SIMPLE = pd.DataFrame.from_records(
    [
        {"test_sample.gender": "Male", "test_sample.interests": "Food", "test_sample_metadata.person_count": 2},
        {"test_sample.gender": "Female", "test_sample.interests": "Sports", "test_sample_metadata.person_count": 5},
        {"test_sample.gender": "Unknown", "test_sample.interests": "Tech", "test_sample_metadata.person_count": 1},
        {
            "test_sample.gender": "Female",
            "test_sample.interests": "Literature",
            "test_sample_metadata.person_count": 10,
        },
        {"test_sample.gender": "Female", "test_sample.interests": "Food", "test_sample_metadata.person_count": 15},
        {"test_sample.gender": "Male", "test_sample.interests": "Tech", "test_sample_metadata.person_count": 11},
        {"test_sample.gender": "Unknown", "test_sample.interests": "Food", "test_sample_metadata.person_count": 8},
        {"test_sample.gender": "Male", "test_sample.interests": "Literature", "test_sample_metadata.person_count": 4},
        {"test_sample.gender": "Female", "test_sample.interests": "Sports", "test_sample_metadata.person_count": 20},
        {"test_sample.gender": "Female", "test_sample.interests": "Tech", "test_sample_metadata.person_count": 3},
    ],
)
DF_SIMPLE["test_sample_id"] = list(range(len(DF_SIMPLE)))
DF_SIMPLE["test_case_id"] = 5777

DF_COMPLEX = pd.DataFrame.from_records(
    [
        {"test_sample.gender": "Male", "test_sample.interests": "Food", "test_sample_metadata.person_count": 2},
        {"test_sample.gender": "Female", "test_sample.interests": "Sports", "test_sample_metadata.person_count": 5},
        {"test_sample.gender": None, "test_sample.interests": "Tech", "test_sample_metadata.person_count": 1},
        {
            "test_sample.gender": "Female",
            "test_sample.interests": "Literature",
            "test_sample_metadata.person_count": None,
        },
        {"test_sample.gender": "Female", "test_sample.interests": "Food", "test_sample_metadata.person_count": 12},
        {"test_sample.gender": "Male", "test_sample.interests": "Tech", "test_sample_metadata.person_count": 11},
        {"test_sample.gender": "Unknown", "test_sample.interests": "Food", "test_sample_metadata.person_count": 8},
        {"test_sample.gender": "Male", "test_sample.interests": "Literature", "test_sample_metadata.person_count": 4},
        {"test_sample.gender": "Female", "test_sample.interests": "Sports", "test_sample_metadata.person_count": 1},
        {"test_sample.gender": "Female", "test_sample.interests": "Tech", "test_sample_metadata.person_count": 3},
    ],
)
DF_COMPLEX["test_sample_id"] = list(range(len(DF_SIMPLE)))
DF_COMPLEX["test_case_id"] = 5900

DIM_INTERESTS = Dimension("test_sample", "interests", CATEGORICAL)
DIM_GENDER = Dimension("test_sample", "gender", CATEGORICAL)
DIM_PERSON_COUNT = Dimension("test_sample_metadata", "person_count", NUMERIC)
INTERESTS_POPULAR = CategoricalBucket(["Sports", "Food"])
INTERESTS_TECH = CategoricalBucket(["Tech"])
SPEC_SIMPLE_CROSS_DIMENSIONS_1 = [
    DimensionSpec(
        "interest",
        [DIM_INTERESTS],
        OrderedDict({"popular": [INTERESTS_POPULAR], "geeky": [INTERESTS_TECH]}),
    ),
    DimensionSpec("gender", [DIM_GENDER], {}),
]
SPEC_SIMPLE_CROSS_DIMENSIONS_2 = [
    DimensionSpec("gender", [DIM_GENDER], {}),
    DimensionSpec(
        "person_count",
        [DIM_PERSON_COUNT],
        OrderedDict({"low": [NumericBucket(0, 5)], "high": [NumericBucket(10, 15, IntervalType.CLOSED)]}),
    ),
]

SPEC_COMBINED_DIMENSIONS = [
    DimensionSpec(
        "interests/gender",
        [DIM_INTERESTS, DIM_GENDER],
        OrderedDict(
            {
                "popular and female": [INTERESTS_POPULAR, CategoricalBucket(["Female"])],
                "geeky and male": [INTERESTS_TECH, CategoricalBucket(["Male"])],
            },
        ),
    ),
]
SPEC_CROSS_DIMENSIONS_WITH_COMBINED = [SPEC_COMBINED_DIMENSIONS[0], SPEC_SIMPLE_CROSS_DIMENSIONS_2[1]]


@pytest.mark.parametrize(
    "df,strata,expected_sample_counts",
    [
        (
            DF_SIMPLE,
            SPEC_SIMPLE_CROSS_DIMENSIONS_1,
            [
                5,  # popular
                3,  # geeky
                5,  # female
                3,  # male
                2,  # unknown
                3,  # popular, female
                1,  # popular, male
                1,  # popular, unknown
                1,  # geeky, female
                1,  # geeky, male
                1,  # geeky, unknown
            ],
        ),
        (DF_SIMPLE, SPEC_SIMPLE_CROSS_DIMENSIONS_2, [5, 3, 2, 4, 3, 1, 2, 2, 1, 1, 0]),
        (DF_SIMPLE, SPEC_COMBINED_DIMENSIONS, [3, 1]),
        (
            DF_SIMPLE,
            SPEC_CROSS_DIMENSIONS_WITH_COMBINED,
            [
                3,  # popular-female
                1,  # geeky-male
                4,  # low,
                3,  # high,
                0,  # popular-female, low
                1,  # popular-female, high
                0,  # geeky-male, low
                1,  # geeky-male, high
            ],
        ),
        (DF_COMPLEX, SPEC_SIMPLE_CROSS_DIMENSIONS_1, [5, 3, 5, 3, 1, 1, 3, 1, 1, 0, 1, 1, 0, 1]),
        (DF_COMPLEX, SPEC_SIMPLE_CROSS_DIMENSIONS_2, [5, 3, 1, 1, 5, 2, 2, 1, 2, 1, 0, 0, 1, 0]),
        (DF_COMPLEX, SPEC_COMBINED_DIMENSIONS, [3, 1]),
        (DF_COMPLEX, SPEC_CROSS_DIMENSIONS_WITH_COMBINED, [3, 1, 5, 2, 1, 1, 0, 1]),
    ],
)
def test_stratification(df: pd.DataFrame, strata: List[DimensionSpec], expected_sample_counts: List[int]) -> None:
    test_suite_name = "My-test !!"
    result = stratify(test_suite_name, "foo", strata, df, omit_empty=False)

    assert result.base_test_case.id == 0
    assert result.base_test_case.sample_count == len(df)
    assert [test_case.sample_count for test_case in result.stratified_test_cases] == expected_sample_counts

    result = stratify(test_suite_name, "foo", strata, df, omit_empty=True)
    expected_non_zero_counts = [count for count in expected_sample_counts if count]
    assert result.base_test_case.id == 0
    assert result.base_test_case.sample_count == len(df)
    assert [test_case.sample_count for test_case in result.stratified_test_cases] == expected_non_zero_counts


def test_dimension_spec_validate_name() -> None:
    dummy = DIM_DUMMY_CATEGORICAL
    with pytest.raises(ValidationError):
        DimensionSpec("", [dummy], {})

    with pytest.raises(ValidationError):
        DimensionSpec("a" * 101, [dummy], {})


@pytest.mark.parametrize(
    "columns,buckets,e",
    [
        ([], {}, ValidationError),
        (
            [
                DIM_DUMMY_CATEGORICAL,
                Dimension("test_sample", "b", CATEGORICAL),
                Dimension("test_sample", "c", CATEGORICAL),
                Dimension("test_sample", "d", CATEGORICAL),
            ],
            {},
            ValidationError,
        ),
        (
            [
                DIM_DUMMY_CATEGORICAL,
                DIM_DUMMY_CATEGORICAL,
            ],
            {"foo": [CategoricalBucket([]), CategoricalBucket([])]},
            ValueError,
        ),
        ([DIM_DUMMY_CATEGORICAL], {"foo": []}, ValueError),
        ([DIM_DUMMY_CATEGORICAL], {"foo": [NumericBucket(0, 10)]}, ValueError),
        ([DIM_DUMMY_NUMERIC], {"foo": [CategoricalBucket([])]}, ValueError),
        ([DIM_DUMMY_NUMERIC], {"foo": [CategoricalBucket([])]}, ValueError),
        ([DIM_DUMMY_NUMERIC], {}, ValueError),
        ([DIM_DUMMY_CATEGORICAL, Dimension("test_sample_metadata", "b", CATEGORICAL)], {}, ValueError),
    ],
)
def test_dimension_spec_validate_columns(columns: List[Dimension], buckets: Dict[str, Any], e: Type[Exception]) -> None:
    with pytest.raises(e):
        DimensionSpec("foo", columns, buckets)


@pytest.mark.parametrize(
    "strata",
    [
        [
            DimensionSpec("a", [DIM_DUMMY_CATEGORICAL], {}),
            DimensionSpec("b", [DIM_DUMMY_CATEGORICAL], {}),
        ],
        [
            DimensionSpec("a", [DIM_DUMMY_CATEGORICAL], {}),
            DimensionSpec("a", [Dimension("test_sample_metadata", "b", CATEGORICAL)], {}),
        ],
    ],
)
def test_stratification_request_validation(strata: List[DimensionSpec]) -> None:
    with pytest.raises(ValueError):
        BaseStratifyRequest("foo", 0, "bar", strata)
