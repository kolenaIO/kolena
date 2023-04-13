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
from collections import OrderedDict
from itertools import chain
from itertools import combinations
from itertools import product
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

import pandas as pd
from pydantic import StrictFloat
from pydantic import StrictStr

from kolena._api.v1.core import CategoricalBucket
from kolena._api.v1.core import Dimension
from kolena._api.v1.core import DimensionSpec
from kolena._api.v1.core import IntervalType
from kolena._api.v1.core import NumericBucket
from kolena._api.v1.core import StratifyResponse
from kolena._api.v1.core import TestCaseInfo

STR_NONE = "null"

T = TypeVar("T")


# https://docs.python.org/3/library/itertools.html#itertools-recipes
def powerset(iterable: Iterable[T]) -> Iterable[Iterable[T]]:
    # powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def get_stratified_test_case_name(test_suite_name: str, dimensions: List[Tuple[str, str]]):
    extension = (
        " :: ".join(f"{field} ({bucket_name})" for field, bucket_name in dimensions) if dimensions else "complete"
    )
    return f"{extension} :: {test_suite_name}"


def get_column_name(dimension: Dimension) -> str:
    return f"{dimension.column}.{dimension.field}"


def bucket_match(x, buckets: List[Union[CategoricalBucket, NumericBucket]]) -> bool:
    return all(single_bucket_match(element, bucket) for element, bucket in zip(x, buckets))


def single_bucket_match(x: Union[StrictStr, StrictFloat], bucket: Union[CategoricalBucket, NumericBucket]) -> bool:
    if isinstance(bucket, CategoricalBucket):
        return x in bucket.values
    else:
        if bucket.interval_type == IntervalType.CLOSED:
            return bucket.min <= x <= bucket.max
        else:
            return bucket.min <= x < bucket.max


def assign_buckets(
    df: pd.DataFrame,
    strata: List[DimensionSpec],
) -> pd.DataFrame:
    base_columns = ["test_sample_id", "test_case_id"]
    df_bucket = pd.DataFrame(columns=base_columns + [s.name for s in strata])
    df_bucket[base_columns] = df[base_columns]
    for stratum in strata:
        fields = [get_column_name(column) for column in stratum.columns]
        # currently only supported auto_stratify is single categorical field, which doesn't need mapping
        if stratum.should_auto_stratify():
            df_bucket[stratum.name] = df[fields]
        else:
            df_bucket[stratum.name] = df[fields].apply(
                lambda x: next(
                    (name for name, conditions in stratum.buckets.items() if bucket_match(x, conditions)),
                    None,
                ),
                axis=1,
            )
    return df_bucket


def fill_buckets(
    request_strata: List[DimensionSpec],
    get_field_values: Callable[[str], List[str]],
) -> List[DimensionSpec]:
    strata = []
    for stratum in request_strata:
        buckets = {**stratum.buckets}
        if len(stratum.columns) == 1 and stratum.columns[0].is_categorical() and not stratum.buckets:
            buckets = OrderedDict(
                (val, [CategoricalBucket(values=[val])])
                if val is not None
                else (STR_NONE, [CategoricalBucket(values=[])])
                for val in get_field_values(stratum.name)
            )
        strata.append(DimensionSpec(name=stratum.name, columns=stratum.columns, buckets=buckets))
    return strata


def iter_stratification(
    strata: List[DimensionSpec],
) -> Iterable[Optional[List[Tuple[str, str]]]]:
    for subset in powerset(strata):
        subset_list = list(subset)
        if not subset_list:
            yield
        else:
            cross_dimensions = list(
                product(
                    *(
                        list((stratum.name, bucket_name) for bucket_name in stratum.buckets.keys())
                        for stratum in subset
                    ),
                ),
            )
            yield from cross_dimensions


def stratify_data(
    test_suite_name: str,
    strata: List[DimensionSpec],
    df: pd.DataFrame,
    omit_empty: bool = True,
) -> Tuple[str, List[Tuple[str, pd.DataFrame, Dict[str, str]]]]:
    """returns (test_suite_name, list[(test_case_name, dataframe, bucket_membership)])"""

    test_cases = []
    df = assign_buckets(df, strata)

    strata = fill_buckets(strata, lambda field: sorted(df[field].unique(), key=lambda x: (x is None, x)))

    for cross in iter_stratification(strata):
        if not cross:
            test_case_name = get_stratified_test_case_name(test_suite_name, [])
            test_cases.append((test_case_name, df, {}))
        else:
            test_case_name = get_stratified_test_case_name(test_suite_name, cross)
            q = " & ".join(
                [
                    f"`{field}`.isnull()" if bucket_name == STR_NONE else f"`{field}` == '{bucket_name}'"
                    for field, bucket_name in cross
                ],
            )
            matched_df = df.query(q)
            if len(matched_df) or not omit_empty:
                test_cases.append((test_case_name, matched_df, dict(cross)))

    return test_suite_name, test_cases


def stratify(
    test_suite_name: str,
    test_suite_description: str,
    strata: List[DimensionSpec],
    df: pd.DataFrame,
    omit_empty: bool = True,
) -> StratifyResponse:
    test_suite_name, test_case_data = stratify_data(test_suite_name, strata, df, omit_empty)
    test_cases = [
        TestCaseInfo(id=0, name=test_case_name, sample_count=len(test_case_df), membership=membership)
        for test_case_name, test_case_df, membership in test_case_data
    ]

    return StratifyResponse(
        test_suite_id=0,
        test_suite_name=test_suite_name,
        test_suite_description=test_suite_description,
        base_test_case=test_cases[0],
        stratified_test_cases=test_cases[1:],
    )
