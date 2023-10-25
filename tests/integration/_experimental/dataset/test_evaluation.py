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
import random
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kolena._experimental.dataset import fetch_evaluation_results
from kolena._experimental.dataset import fetch_inferences
from kolena._experimental.dataset import register_dataset
from kolena._experimental.dataset import test
from kolena._experimental.dataset._evaluation import EVAL_FUNC_TYPE
from kolena._experimental.dataset._evaluation import INFER_FUNC_TYPE
from kolena._experimental.dataset._evaluation import TYPE_EVALUATION_CONFIG
from kolena.errors import IncorrectUsageError
from kolena.errors import NotFoundError
from tests.integration.helper import fake_locator
from tests.integration.helper import with_test_prefix


def _assert_frame_equal(df1: pd.DataFrame, df2: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
    """wrapper of assert_frame_equal with selected columns options"""
    if columns is None:
        assert_frame_equal(df1, df2)
    else:
        assert_frame_equal(df1[columns], df2[columns])


def get_df_dp(n: int = 20) -> pd.DataFrame:
    records = [
        dict(
            user_dp_id=i,
            locator=fake_locator(i, "datapoints"),
            width=i + 500,
            height=i + 400,
            city=random.choice(["new york", "waterloo"]),
        )
        for i in range(n)
    ]
    return pd.DataFrame(records)


def get_df_inf(n: int = 20) -> pd.DataFrame:
    records = [dict(user_dp_id=i, softmax_bitmap=fake_locator(i, "softmax_bitmap")) for i in range(n)]
    return pd.DataFrame(records)


def get_df_mtr(n: int = 20) -> pd.DataFrame:
    records = [dict(user_dp_id=i, score=i) for i in range(n)]
    return pd.DataFrame(records)


def get_infer_func(
    df_inf: pd.DataFrame,
    columns: List[str],
    id_col: str = "user_dp_id",
    how: str = "left",
    keep_none: bool = False,
) -> INFER_FUNC_TYPE:
    def infer_func(datapoints: pd.DataFrame) -> pd.DataFrame:
        _inf = datapoints.set_index(id_col).join(df_inf.set_index(id_col), how=how).reset_index()[columns]
        if keep_none:
            _inf = _inf.replace({np.nan: None})
        return _inf

    return infer_func


def get_eval_func(
    df_mtr: pd.DataFrame,
    columns: List[str],
    id_col: str = "user_dp_id",
    how: str = "left",
    keep_none: bool = False,
) -> EVAL_FUNC_TYPE:
    def eval_func(
        datapoints: pd.DataFrame,
        inferences: pd.DataFrame,
        eval_config: TYPE_EVALUATION_CONFIG,
    ) -> pd.DataFrame:
        _metrics = datapoints.set_index(id_col).join(df_mtr.set_index(id_col), how=how).reset_index()[columns]
        if keep_none:
            _metrics = _metrics.replace({np.nan: None})
        return _metrics

    return eval_func


def test__test() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__test")
    model_name = with_test_prefix(f"{__file__}::test__test")
    df_dp = get_df_dp()
    dp_columns = ["user_dp_id", "locator", "width", "height", "city"]
    register_dataset(dataset_name, df_dp[3:10][dp_columns])

    df_inf = get_df_inf()
    df_mtr = get_df_mtr()
    inf_columns = ["softmax_bitmap"]
    mtr_columns = ["score"]

    eval_configs = [
        {"threshold": 0.3},
        {"threshold": 0.6},
    ]
    test(
        dataset_name,
        model_name,
        infer=get_infer_func(df_inf, inf_columns),
        eval=get_eval_func(df_mtr, mtr_columns),
        eval_configs=eval_configs,
    )

    df_by_eval = fetch_evaluation_results(dataset_name, model_name)
    assert len(df_by_eval) == 2
    assert sorted([cfg for cfg, *_ in df_by_eval], key=lambda x: x.get("threshold")) == sorted(
        eval_configs,
        key=lambda x: x.get("threshold"),
    )
    expected_df_dp = df_dp[3:10].reset_index(drop=True)
    expected_df_inf = df_inf[3:10].reset_index(drop=True)
    expected_df_mtr = df_mtr[3:10].reset_index(drop=True)
    for df_eval in df_by_eval:
        _assert_frame_equal(df_eval[1], expected_df_dp, dp_columns)
        _assert_frame_equal(df_eval[2], expected_df_inf, inf_columns)
        _assert_frame_equal(df_eval[3], expected_df_mtr, mtr_columns)


def test__test__df_for_infer_and_eval() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__test__df_for_infer_and_eval")
    model_name = with_test_prefix(f"{__file__}::test__test__df_for_infer_and_eval")
    df_dp = get_df_dp()
    dp_columns = ["user_dp_id", "locator", "width", "height", "city"]
    register_dataset(dataset_name, df_dp[3:10][dp_columns])

    df_inf = get_df_inf()
    df_mtr = get_df_mtr()
    inf_columns = ["softmax_bitmap"]
    mtr_columns = ["score"]

    test(
        dataset_name,
        model_name,
        infer=df_inf,
        eval=df_mtr,
        on="user_dp_id",
    )

    df_by_eval = fetch_evaluation_results(dataset_name, model_name)
    assert len(df_by_eval) == 1
    expected_df_dp = df_dp[3:10].reset_index(drop=True)
    expected_df_inf = df_inf[3:10].reset_index(drop=True)
    expected_df_mtr = df_mtr[3:10].reset_index(drop=True)
    for df_eval in df_by_eval:
        _assert_frame_equal(df_eval[1], expected_df_dp, dp_columns)
        _assert_frame_equal(df_eval[2], expected_df_inf, inf_columns)
        _assert_frame_equal(df_eval[3], expected_df_mtr, mtr_columns)


def test__test__missing_inference() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__test__missing_inference")
    model_name = with_test_prefix(f"{__file__}::test__test__missing_inference")
    df_dp = get_df_dp()
    dp_columns = ["user_dp_id", "locator", "width", "height", "city"]
    register_dataset(dataset_name, df_dp[3:10][dp_columns])

    df_inf = get_df_inf(5)
    inf_columns = ["softmax_bitmap"]

    test(
        dataset_name,
        model_name,
        infer=get_infer_func(df_inf, inf_columns),
    )

    df_datapoints, df_inferences = fetch_inferences(dataset_name, model_name)
    expected_df_dp = df_dp[3:10].reset_index(drop=True)
    expected_df_inf = pd.concat([df_inf[3:5], pd.DataFrame({"softmax_bitmap": [np.nan] * 5})], axis=0).reset_index(
        drop=True,
    )
    _assert_frame_equal(df_datapoints, expected_df_dp, dp_columns)
    _assert_frame_equal(df_inferences, expected_df_inf, inf_columns)

    # add 3 new datapoints, then we should have missing inference in the db records
    register_dataset(dataset_name, df_dp[:10][dp_columns])
    df_datapoints, df_inferences = fetch_inferences(dataset_name, model_name)
    expected_df_dp = pd.concat([df_dp[3:10], df_dp[:3]]).reset_index(drop=True)
    expected_df_inf = pd.concat(
        [df_inf[3:5], pd.DataFrame({"softmax_bitmap": [np.nan] * (5 + 3)})],
        axis=0,
    ).reset_index(drop=True)
    _assert_frame_equal(df_datapoints, expected_df_dp, dp_columns)
    _assert_frame_equal(df_inferences, expected_df_inf, inf_columns)


def test__test__missing_metrics() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__test__missing_metrics")
    model_name = with_test_prefix(f"{__file__}::test__test__missing_metrics")
    df_dp = get_df_dp()
    dp_columns = ["user_dp_id", "locator", "width", "height", "city"]
    register_dataset(dataset_name, df_dp[13:20][dp_columns])

    df_inf = get_df_inf()
    df_mtr = get_df_mtr()
    inf_columns = ["softmax_bitmap"]
    mtr_columns = ["score"]

    test(
        dataset_name,
        model_name,
        infer=get_infer_func(df_inf, inf_columns),
        eval=get_eval_func(df_mtr, mtr_columns),
    )

    df_by_eval = fetch_evaluation_results(dataset_name, model_name)
    eval_cfg, df_datapoints, df_inferences, df_metrics = df_by_eval[0]
    expected_df_dp = df_dp[13:20].reset_index(drop=True)
    expected_df_inf = pd.concat([df_inf[13:20]], axis=0).reset_index(
        drop=True,
    )
    expected_df_mtr = pd.concat([df_mtr[13:20]], axis=0).reset_index(
        drop=True,
    )
    assert len(df_by_eval) == 1
    assert eval_cfg is None
    _assert_frame_equal(df_datapoints, expected_df_dp, dp_columns)
    _assert_frame_equal(df_inferences, expected_df_inf, inf_columns)
    _assert_frame_equal(df_metrics, expected_df_mtr, mtr_columns)

    # add 3 new datapoints, then we should have missing inference and metrics in the db records
    register_dataset(dataset_name, df_dp[10:20][dp_columns])

    df_by_eval = fetch_evaluation_results(dataset_name, model_name)
    eval_cfg, df_datapoints, df_inferences, df_metrics = df_by_eval[0]
    expected_df_dp = pd.concat([df_dp[13:20], df_dp[10:13]]).reset_index(drop=True)
    expected_df_inf = pd.concat(
        [df_inf[13:20], pd.DataFrame({"softmax_bitmap": [np.nan] * 3})],
        axis=0,
    ).reset_index(drop=True)
    expected_df_mtr = pd.concat([df_mtr[13:20], pd.DataFrame({"score": [np.nan] * 3})], axis=0).reset_index(
        drop=True,
    )
    assert len(df_by_eval) == 1
    assert eval_cfg is None
    _assert_frame_equal(df_datapoints, expected_df_dp, dp_columns)
    _assert_frame_equal(df_inferences, expected_df_inf, inf_columns)
    _assert_frame_equal(df_metrics, expected_df_mtr, mtr_columns)

    with pytest.raises(IncorrectUsageError) as exc_info:
        test(
            dataset_name,
            model_name,
            eval=get_eval_func(df_mtr, mtr_columns),
        )
    exc_info_value = str(exc_info.value)
    assert "cannot upload metrics without inference" in exc_info_value


def test__test__upload_none() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__test__upload_none")
    model_name = with_test_prefix(f"{__file__}::test__test__upload_none")
    df_dp = get_df_dp()
    dp_columns = ["user_dp_id", "locator", "width", "height", "city"]
    register_dataset(dataset_name, df_dp[dp_columns])

    df_inf = get_df_inf(10)
    df_mtr = get_df_mtr(10)
    inf_columns = ["softmax_bitmap"]
    mtr_columns = ["score"]

    test(
        dataset_name,
        model_name,
        infer=get_infer_func(df_inf, inf_columns, keep_none=True),
        eval=get_eval_func(df_mtr, mtr_columns, keep_none=True),
    )

    df_by_eval = fetch_evaluation_results(dataset_name, model_name)
    eval_cfg, df_datapoints, df_inferences, df_metrics = df_by_eval[0]
    expected_df_dp = df_dp.reset_index(drop=True)
    expected_df_inf = pd.concat([df_inf, pd.DataFrame({"softmax_bitmap": [np.nan] * 10})], axis=0).reset_index(
        drop=True,
    )
    expected_df_mtr = pd.concat([df_mtr, pd.DataFrame({"score": [np.nan] * 10})], axis=0).reset_index(
        drop=True,
    )
    assert len(df_by_eval) == 1
    assert eval_cfg is None
    _assert_frame_equal(df_datapoints, expected_df_dp, dp_columns)
    _assert_frame_equal(df_inferences, expected_df_inf, inf_columns)
    _assert_frame_equal(df_metrics, expected_df_mtr, mtr_columns)


def test__test__invalid_data__eval_before_inf() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__test__invalid_data__eval_before_inf")
    model_name = with_test_prefix(f"{__file__}::test__test__invalid_data__eval_before_inf")
    df_dp = get_df_dp()
    dp_columns = ["user_dp_id", "locator", "width", "height", "city"]
    register_dataset(dataset_name, df_dp[13:20][dp_columns])

    df_mtr = get_df_mtr()
    mtr_columns = ["score"]

    with pytest.raises(IncorrectUsageError) as exc_info:
        test(
            dataset_name,
            model_name,
            eval=get_eval_func(df_mtr, mtr_columns),
        )
    exc_info_value = str(exc_info.value)
    assert "cannot upload metrics without inference" in exc_info_value


def test__test__invalid_data__df_size_mismatch() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__test__invalid_data__df_size_mismatch")
    model_name = with_test_prefix(f"{__file__}::test__test__invalid_data__df_size_mismatch")
    df_dp = get_df_dp()
    dp_columns = ["user_dp_id", "locator", "width", "height", "city"]
    register_dataset(dataset_name, df_dp[dp_columns])

    df_inf = get_df_inf(10)
    df_mtr = get_df_mtr(10)
    inf_columns = ["softmax_bitmap"]
    mtr_columns = ["score"]

    with pytest.raises(IncorrectUsageError) as exc_info:
        test(
            dataset_name,
            model_name,
            infer=get_infer_func(df_inf, inf_columns, how="inner"),
        )
    exc_info_value = str(exc_info.value)
    assert "numbers of rows between two dataframe do not match" in exc_info_value

    with pytest.raises(IncorrectUsageError) as exc_info:
        test(
            dataset_name,
            model_name,
            eval=get_eval_func(df_mtr, mtr_columns, how="inner"),
        )
    exc_info_value = str(exc_info.value)
    assert "numbers of rows between two dataframe do not match" in exc_info_value


def test__fetch_inferences__not_exist() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__fetch_inferences__not_exist")
    model_name = with_test_prefix(f"{__file__}::test__fetch_inferences__not_exist")
    df_datapoints, df_inferences = fetch_inferences(dataset_name, model_name)
    assert df_datapoints.empty
    assert df_inferences.empty


def test__fetch_evaluation_results__not_exist() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__fetch_evaluation_results__not_exist")
    model_name = with_test_prefix(f"{__file__}::test__fetch_evaluation_results__not_exist")
    with pytest.raises(NotFoundError):
        fetch_evaluation_results(dataset_name, model_name)
