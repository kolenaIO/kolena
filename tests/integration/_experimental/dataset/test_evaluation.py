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

from kolena._experimental.dataset import fetch_dataset
from kolena._experimental.dataset import fetch_results
from kolena._experimental.dataset import register_dataset
from kolena._experimental.dataset import test
from kolena.errors import IncorrectUsageError
from kolena.errors import NotFoundError
from tests.integration._experimental.dataset.test_dataset import batch_iterator
from tests.integration.helper import fake_locator
from tests.integration.helper import with_test_prefix

JOIN_COLUMN = "user_dp_id"
ID_FIELDS = [JOIN_COLUMN]


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


def get_df_result(n: int = 20) -> pd.DataFrame:
    records = [dict(user_dp_id=i, softmax_bitmap=fake_locator(i, "softmax_bitmap"), score=i * 0.1) for i in range(n)]
    return pd.DataFrame(records)


def test__test() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__test")
    model_name = with_test_prefix(f"{__file__}::test__test")
    df_dp = get_df_dp()
    dp_columns = [JOIN_COLUMN, "locator", "width", "height", "city"]
    register_dataset(dataset_name, df_dp[3:10][dp_columns], id_fields=ID_FIELDS)

    df_result = get_df_result()
    result_columns = ["softmax_bitmap", "score"]
    test(
        dataset_name,
        model_name,
        df_result,
    )

    fetched_df_dp, df_results_by_eval = fetch_results(dataset_name, model_name)
    eval_cfg, fetched_df_result = df_results_by_eval[0]
    assert len(df_results_by_eval) == 1
    assert eval_cfg is None
    expected_df_dp = df_dp[3:10].reset_index(drop=True)
    expected_df_result = df_result.drop(columns=[JOIN_COLUMN])[3:10].reset_index(drop=True)
    _assert_frame_equal(fetched_df_dp, expected_df_dp, dp_columns)
    _assert_frame_equal(fetched_df_result, expected_df_result, result_columns)


def test__test__iterator_input() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__test__iterator_input")
    model_name = with_test_prefix(f"{__file__}::test__test__iterator_input")
    df_dp = get_df_dp()
    dp_columns = [JOIN_COLUMN, "locator", "width", "height", "city"]
    register_dataset(dataset_name, df_dp[3:10][dp_columns], id_fields=ID_FIELDS)

    df_result = get_df_result()
    df_result_iterator = batch_iterator(df_result)
    result_columns = ["softmax_bitmap", "score"]

    test(
        dataset_name,
        model_name,
        df_result_iterator,
    )

    fetched_df_dp, df_results_by_eval = fetch_results(dataset_name, model_name)
    eval_cfg, fetched_df_result = df_results_by_eval[0]
    assert len(df_results_by_eval) == 1
    assert eval_cfg is None
    expected_df_dp = df_dp[3:10].reset_index(drop=True)
    expected_df_result = df_result.drop(columns=[JOIN_COLUMN])[3:10].reset_index(drop=True)
    _assert_frame_equal(fetched_df_dp, expected_df_dp, dp_columns)
    _assert_frame_equal(fetched_df_result, expected_df_result, result_columns)


def test__test__align_manually() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__test__align_manually")
    model_name = with_test_prefix(f"{__file__}::test__test__align_manually")
    df_dp = get_df_dp()
    dp_columns = [JOIN_COLUMN, "locator", "width", "height", "city"]
    register_dataset(dataset_name, df_dp[3:10][dp_columns], id_fields=ID_FIELDS)

    fetched_df_dp = fetch_dataset(dataset_name)
    df_result = get_df_result()
    result_columns = ["softmax_bitmap", "score"]
    aligned_df_result = fetched_df_dp[[JOIN_COLUMN]].merge(df_result, how="left", on=JOIN_COLUMN)

    test(
        dataset_name,
        model_name,
        aligned_df_result,
    )

    fetched_df_dp, df_results_by_eval = fetch_results(dataset_name, model_name)
    eval_cfg, fetched_df_result = df_results_by_eval[0]
    assert len(df_results_by_eval) == 1
    assert eval_cfg is None
    expected_df_dp = df_dp[3:10].reset_index(drop=True)
    expected_df_result = df_result.drop(columns=[JOIN_COLUMN])[3:10].reset_index(drop=True)
    _assert_frame_equal(fetched_df_dp, expected_df_dp, dp_columns)
    _assert_frame_equal(fetched_df_result, expected_df_result, result_columns)


def test__test__multiple_eval_configs() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__test__multiple_eval_configs")
    model_name = with_test_prefix(f"{__file__}::test__test__multiple_eval_configs")
    df_dp = get_df_dp()
    dp_columns = [JOIN_COLUMN, "locator", "width", "height", "city"]
    register_dataset(dataset_name, df_dp[3:10][dp_columns], id_fields=ID_FIELDS)

    df_result = get_df_result()
    input_result_columns_1 = [JOIN_COLUMN, "softmax_bitmap", "score"]
    input_result_columns_2 = [JOIN_COLUMN, "softmax_bitmap"]
    result_columns_1 = input_result_columns_1[1:]
    result_columns_2 = input_result_columns_2[1:]
    df_result_1 = df_result[input_result_columns_1]
    df_result_2 = df_result[input_result_columns_2]
    eval_config_1 = dict(threshold=0.1)
    eval_config_2 = dict(threshold=0.2)

    test(
        dataset_name,
        model_name,
        [(eval_config_1, df_result_1), (eval_config_2, df_result_2)],
    )

    fetched_df_dp, df_results_by_eval = fetch_results(dataset_name, model_name)
    assert len(df_results_by_eval) == 2
    expected_df_dp = df_dp[3:10].reset_index(drop=True)
    _assert_frame_equal(fetched_df_dp, expected_df_dp, dp_columns)

    df_results_by_eval = sorted(df_results_by_eval, key=lambda x: x[0].get("threshold"))
    fetched_eval_config_1, fetched_df_result_1 = df_results_by_eval[0]
    fetched_eval_config_2, fetched_df_result_2 = df_results_by_eval[1]
    assert fetched_eval_config_1 == eval_config_1
    assert fetched_eval_config_2 == eval_config_2
    expected_df_result_1 = df_result_1.drop(columns=[JOIN_COLUMN])[3:10].reset_index(drop=True)
    expected_df_result_2 = df_result_2.drop(columns=[JOIN_COLUMN])[3:10].reset_index(drop=True)
    _assert_frame_equal(fetched_df_result_1, expected_df_result_1, result_columns_1)
    _assert_frame_equal(fetched_df_result_2, expected_df_result_2, result_columns_2)


def test__test__multiple_eval_configs__iterator_input() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__test__multiple_eval_configs__iterator_input")
    model_name = with_test_prefix(f"{__file__}::test__test__multiple_eval_configs__iterator_input")
    df_dp = get_df_dp()
    dp_columns = [JOIN_COLUMN, "locator", "width", "height", "city"]
    register_dataset(dataset_name, df_dp[3:10][dp_columns], id_fields=[JOIN_COLUMN])

    df_result = get_df_result()
    result_columns_1 = [JOIN_COLUMN, "softmax_bitmap", "score"]
    result_columns_2 = [JOIN_COLUMN, "softmax_bitmap"]
    df_result_1 = df_result[result_columns_1]
    df_result_1_iterator = batch_iterator(df_result_1)
    df_result_2 = df_result[result_columns_2]
    df_result_2_iterator = batch_iterator(df_result_2)
    eval_config_1 = dict(threshold=0.1)
    eval_config_2 = dict(threshold=0.2)

    test(
        dataset_name,
        model_name,
        [(eval_config_1, df_result_1_iterator), (eval_config_2, df_result_2_iterator)],
    )

    fetched_df_dp, df_results_by_eval = fetch_results(dataset_name, model_name)
    assert len(df_results_by_eval) == 2
    expected_df_dp = df_dp[3:10].reset_index(drop=True)
    _assert_frame_equal(fetched_df_dp, expected_df_dp, dp_columns)
    df_results_by_eval = sorted(df_results_by_eval, key=lambda x: x[0].get("threshold"))
    fetched_eval_config_1, fetched_df_result_1 = df_results_by_eval[0]
    fetched_eval_config_2, fetched_df_result_2 = df_results_by_eval[1]
    assert fetched_eval_config_1 == eval_config_1
    assert fetched_eval_config_2 == eval_config_2
    expected_df_result_1 = df_result_1.drop(columns=[JOIN_COLUMN])[3:10].reset_index(drop=True)
    expected_df_result_2 = df_result_2.drop(columns=[JOIN_COLUMN])[3:10].reset_index(drop=True)
    result_columns_1.remove(JOIN_COLUMN)
    result_columns_2.remove(JOIN_COLUMN)
    _assert_frame_equal(fetched_df_result_1, expected_df_result_1, result_columns_1)
    _assert_frame_equal(fetched_df_result_2, expected_df_result_2, result_columns_2)


def test__test__multiple_eval_configs__partial_uploading() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__test__multiple_eval_configs__partial_uploading")
    model_name = with_test_prefix(f"{__file__}::test__test__multiple_eval_configs__partial_uploading")
    df_dp = get_df_dp(10)
    dp_columns = [JOIN_COLUMN, "locator", "width", "height", "city"]
    register_dataset(dataset_name, df_dp[dp_columns], id_fields=ID_FIELDS)

    df_result = get_df_result(10)
    input_result_columns_1 = [JOIN_COLUMN, "softmax_bitmap", "score"]
    input_result_columns_2 = [JOIN_COLUMN, "softmax_bitmap"]
    result_columns_1 = input_result_columns_1[1:]
    result_columns_2 = input_result_columns_2[1:]
    df_result_1_p1 = df_result[:5][input_result_columns_1]
    df_result_2_p1 = df_result[5:10][input_result_columns_2]
    eval_config_1 = dict(threshold=0.1)
    eval_config_2 = dict(threshold=0.2)

    test(
        dataset_name,
        model_name,
        [(eval_config_1, df_result_1_p1), (eval_config_2, df_result_2_p1)],
    )

    df_result_1_p2 = df_result[5:10][input_result_columns_1]
    df_result_2_p2 = df_result[:5][input_result_columns_2]
    test(
        dataset_name,
        model_name,
        [(eval_config_1, df_result_1_p2), (eval_config_2, df_result_2_p2)],
    )

    fetched_df_dp, df_results_by_eval = fetch_results(dataset_name, model_name)
    assert len(df_results_by_eval) == 2
    expected_df_dp = df_dp.reset_index(drop=True)
    _assert_frame_equal(fetched_df_dp, expected_df_dp, dp_columns)

    df_results_by_eval = sorted(df_results_by_eval, key=lambda x: x[0].get("threshold"))
    fetched_eval_config_1, fetched_df_result_1 = df_results_by_eval[0]
    fetched_eval_config_2, fetched_df_result_2 = df_results_by_eval[1]
    assert fetched_eval_config_1 == eval_config_1
    assert fetched_eval_config_2 == eval_config_2
    expected_df_result_1 = df_result.drop(columns=[JOIN_COLUMN])[result_columns_1].reset_index(drop=True)
    expected_df_result_2 = df_result.drop(columns=[JOIN_COLUMN])[result_columns_2].reset_index(drop=True)
    _assert_frame_equal(fetched_df_result_1, expected_df_result_1, result_columns_1)
    _assert_frame_equal(fetched_df_result_2, expected_df_result_2, result_columns_2)


def test__test__multiple_eval_configs__duplicate() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__test__multiple_eval_configs__duplicate")
    model_name = with_test_prefix(f"{__file__}::test__test__multiple_eval_configs__duplicate")
    df_dp = get_df_dp()
    dp_columns = [JOIN_COLUMN, "locator", "width", "height", "city"]
    register_dataset(dataset_name, df_dp[3:10][dp_columns], id_fields=ID_FIELDS)

    df_result = get_df_result()
    result_columns_1 = [JOIN_COLUMN, "softmax_bitmap", "score"]
    result_columns_2 = [JOIN_COLUMN, "softmax_bitmap"]
    df_result_1 = df_result[result_columns_1]
    df_result_2 = df_result[result_columns_2]
    eval_config = dict(threshold=0.1)

    with pytest.raises(IncorrectUsageError) as exc_info:
        test(
            dataset_name,
            model_name,
            [(eval_config, df_result_1), (eval_config, df_result_2)],
        )

    exc_info_value = str(exc_info.value)
    assert "duplicate eval configs are invalid" in exc_info_value


def test__test__missing_result() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__test__missing_result")
    model_name = with_test_prefix(f"{__file__}::test__test__missing_result")
    df_dp = get_df_dp()
    dp_columns = [JOIN_COLUMN, "locator", "width", "height", "city"]
    register_dataset(dataset_name, df_dp[3:10][dp_columns], id_fields=ID_FIELDS)

    df_result = get_df_result()
    result_columns = ["softmax_bitmap", "score"]

    test(
        dataset_name,
        model_name,
        df_result,
    )

    fetched_df_dp, df_results_by_eval = fetch_results(dataset_name, model_name)
    eval_cfg, fetched_df_result = df_results_by_eval[0]
    assert len(df_results_by_eval) == 1
    assert eval_cfg is None
    expected_df_dp = df_dp[3:10].reset_index(drop=True)
    expected_df_result = df_result.drop(columns=[JOIN_COLUMN])[3:10].reset_index(drop=True)
    _assert_frame_equal(fetched_df_dp, expected_df_dp, dp_columns)
    _assert_frame_equal(fetched_df_result, expected_df_result, result_columns)

    # add 3 new datapoints, then we should have missing results in the db records
    register_dataset(dataset_name, df_dp[:10][dp_columns], id_fields=ID_FIELDS)
    fetched_df_dp, df_results_by_eval = fetch_results(dataset_name, model_name)
    eval_cfg, fetched_df_result = df_results_by_eval[0]
    assert len(df_results_by_eval) == 1
    assert eval_cfg is None
    expected_df_dp = pd.concat([df_dp[3:10], df_dp[:3]]).sort_values(JOIN_COLUMN).reset_index(drop=True)
    expected_df_result = pd.concat(
        [
            df_result.drop(columns=[JOIN_COLUMN])[3:10],
            pd.DataFrame({"softmax_bitmap": [np.nan] * 3, "score": [np.nan] * 3}),
        ],
        axis=0,
    ).reset_index(drop=True)
    fetched_df_dp = fetched_df_dp.sort_values(JOIN_COLUMN)
    expected_df_dp = expected_df_dp.sort_values(JOIN_COLUMN)
    fetched_df_dp.reset_index(drop=True, inplace=True)
    expected_df_dp.reset_index(drop=True, inplace=True)
    _assert_frame_equal(fetched_df_dp, expected_df_dp, dp_columns)
    _assert_frame_equal(fetched_df_result, expected_df_result, result_columns)


def test__test__upload_none() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__test__upload_none")
    model_name = with_test_prefix(f"{__file__}::test__test__upload_none")
    df_dp = get_df_dp()
    dp_columns = [JOIN_COLUMN, "locator", "width", "height", "city"]
    register_dataset(dataset_name, df_dp[dp_columns], id_fields=ID_FIELDS)

    df_result = get_df_result(10)
    result_columns = ["softmax_bitmap", "score"]

    test(
        dataset_name,
        model_name,
        df_result,
    )

    fetched_df_dp, df_results_by_eval = fetch_results(dataset_name, model_name)
    eval_cfg, fetched_df_result = df_results_by_eval[0]
    expected_df_dp = df_dp.reset_index(drop=True)
    expected_df_result = pd.concat(
        [df_result.drop(columns=JOIN_COLUMN), pd.DataFrame({"softmax_bitmap": [np.nan] * 10, "score": [np.nan] * 10})],
        axis=0,
    ).reset_index(
        drop=True,
    )
    assert len(df_results_by_eval) == 1
    assert eval_cfg is None
    _assert_frame_equal(fetched_df_dp, expected_df_dp, dp_columns)
    _assert_frame_equal(fetched_df_result, expected_df_result, result_columns)


def test__fetch_results__not_exist() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__fetch_results__not_exist")
    model_name = with_test_prefix(f"{__file__}::test__fetch_results__not_exist")
    with pytest.raises(NotFoundError) as exc_info:
        fetch_results(dataset_name, model_name)
    exc_info_value = str(exc_info.value)
    assert "no such model" in exc_info_value
