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
import random

import numpy as np
import pandas as pd
import pytest

from kolena.dataset import download_dataset
from kolena.dataset import download_results
from kolena.dataset import upload_dataset
from kolena.dataset.evaluation import _upload_results
from kolena.errors import IncorrectUsageError
from kolena.errors import NotFoundError
from tests.integration.dataset.test_dataset import batch_iterator
from tests.integration.helper import assert_frame_equal
from tests.integration.helper import fake_locator
from tests.integration.helper import with_test_prefix

JOIN_COLUMN = "user_dp_id"
ID_FIELDS = [JOIN_COLUMN]


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


def test__upload_results() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__upload_results")
    model_name = with_test_prefix(f"{__file__}::test__upload_results")
    df_dp = get_df_dp()
    dp_columns = [JOIN_COLUMN, "locator", "width", "height", "city"]
    upload_dataset(dataset_name, df_dp[3:10][dp_columns], id_fields=ID_FIELDS)

    df_result = get_df_result()
    result_columns = ["softmax_bitmap", "score"]
    response = _upload_results(
        dataset_name,
        model_name,
        df_result,
    )
    assert response.n_inserted == 7
    assert response.n_updated == 0

    fetched_df_dp, df_results_by_eval = download_results(dataset_name, model_name)
    eval_cfg, fetched_df_result = df_results_by_eval[0]
    assert len(df_results_by_eval) == 1
    assert eval_cfg is None
    expected_df_dp = df_dp[3:10].reset_index(drop=True)
    expected_df_result = df_result.drop(columns=[JOIN_COLUMN])[3:10].reset_index(drop=True)
    assert_frame_equal(fetched_df_dp, expected_df_dp, dp_columns)
    assert_frame_equal(fetched_df_result, expected_df_result, result_columns)


def test__upload_results__iterator_input() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__upload_results__iterator_input")
    model_name = with_test_prefix(f"{__file__}::test__upload_results__iterator_input")
    df_dp = get_df_dp()
    dp_columns = [JOIN_COLUMN, "locator", "width", "height", "city"]
    upload_dataset(dataset_name, df_dp[3:10][dp_columns], id_fields=ID_FIELDS)

    df_result = get_df_result()
    df_result_iterator = batch_iterator(df_result)
    result_columns = ["softmax_bitmap", "score"]

    response = _upload_results(dataset_name, model_name, df_result_iterator)
    assert response.n_inserted == 7
    assert response.n_updated == 0

    fetched_df_dp, df_results_by_eval = download_results(dataset_name, model_name)
    eval_cfg, fetched_df_result = df_results_by_eval[0]
    assert len(df_results_by_eval) == 1
    assert eval_cfg is None
    expected_df_dp = df_dp[3:10].reset_index(drop=True)
    expected_df_result = df_result.drop(columns=[JOIN_COLUMN])[3:10].reset_index(drop=True)
    assert_frame_equal(fetched_df_dp, expected_df_dp, dp_columns)
    assert_frame_equal(fetched_df_result, expected_df_result, result_columns)


def test__upload_results__align_manually() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__upload_results__align_manually")
    model_name = with_test_prefix(f"{__file__}::test__upload_results__align_manually")
    df_dp = get_df_dp()
    dp_columns = [JOIN_COLUMN, "locator", "width", "height", "city"]
    upload_dataset(dataset_name, df_dp[3:10][dp_columns], id_fields=ID_FIELDS)

    fetched_df_dp = download_dataset(dataset_name)
    df_result = get_df_result()
    result_columns = ["softmax_bitmap", "score"]
    aligned_df_result = fetched_df_dp[[JOIN_COLUMN]].merge(df_result, how="left", on=JOIN_COLUMN)

    response = _upload_results(
        dataset_name,
        model_name,
        aligned_df_result,
    )
    assert response.n_inserted == 7
    assert response.n_updated == 0

    fetched_df_dp, df_results_by_eval = download_results(dataset_name, model_name)
    eval_cfg, fetched_df_result = df_results_by_eval[0]
    assert len(df_results_by_eval) == 1
    assert eval_cfg is None
    expected_df_dp = df_dp[3:10].reset_index(drop=True)
    expected_df_result = df_result.drop(columns=[JOIN_COLUMN])[3:10].reset_index(drop=True)
    assert_frame_equal(fetched_df_dp, expected_df_dp, dp_columns)
    assert_frame_equal(fetched_df_result, expected_df_result, result_columns)


def test__upload_results__multiple_eval_configs() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__upload_results__multiple_eval_configs")
    model_name = with_test_prefix(f"{__file__}::test__upload_results__multiple_eval_configs")
    df_dp = get_df_dp()
    dp_columns = [JOIN_COLUMN, "locator", "width", "height", "city"]
    upload_dataset(dataset_name, df_dp[3:10][dp_columns], id_fields=ID_FIELDS)

    df_result = get_df_result()
    input_result_columns_1 = [JOIN_COLUMN, "softmax_bitmap", "score"]
    input_result_columns_2 = [JOIN_COLUMN, "softmax_bitmap"]
    result_columns_1 = input_result_columns_1[1:]
    result_columns_2 = input_result_columns_2[1:]
    df_result_1 = df_result[input_result_columns_1]
    df_result_2 = df_result[input_result_columns_2]
    eval_config_1 = dict(threshold=0.1)
    eval_config_2 = dict(threshold=0.2)

    response = _upload_results(
        dataset_name,
        model_name,
        [(eval_config_1, df_result_1), (eval_config_2, df_result_2)],
    )
    assert response.n_inserted == 14
    assert response.n_updated == 0

    fetched_df_dp, df_results_by_eval = download_results(dataset_name, model_name)
    assert len(df_results_by_eval) == 2
    expected_df_dp = df_dp[3:10].reset_index(drop=True)
    assert_frame_equal(fetched_df_dp, expected_df_dp, dp_columns)

    df_results_by_eval = sorted(df_results_by_eval, key=lambda x: x[0].get("threshold"))
    fetched_eval_config_1, fetched_df_result_1 = df_results_by_eval[0]
    fetched_eval_config_2, fetched_df_result_2 = df_results_by_eval[1]
    assert fetched_eval_config_1 == eval_config_1
    assert fetched_eval_config_2 == eval_config_2
    expected_df_result_1 = df_result_1.drop(columns=[JOIN_COLUMN])[3:10].reset_index(drop=True)
    expected_df_result_2 = df_result_2.drop(columns=[JOIN_COLUMN])[3:10].reset_index(drop=True)
    assert_frame_equal(fetched_df_result_1, expected_df_result_1, result_columns_1)
    assert_frame_equal(fetched_df_result_2, expected_df_result_2, result_columns_2)


def test__upload_results__multiple_eval_configs__iterator_input() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__upload_results__multiple_eval_configs__iterator_input")
    model_name = with_test_prefix(f"{__file__}::test__upload_results__multiple_eval_configs__iterator_input")
    df_dp = get_df_dp()
    dp_columns = [JOIN_COLUMN, "locator", "width", "height", "city"]
    upload_dataset(dataset_name, df_dp[3:10][dp_columns], id_fields=[JOIN_COLUMN])

    df_result = get_df_result()
    result_columns_1 = [JOIN_COLUMN, "softmax_bitmap", "score"]
    result_columns_2 = [JOIN_COLUMN, "softmax_bitmap"]
    df_result_1 = df_result[result_columns_1]
    df_result_1_iterator = batch_iterator(df_result_1)
    df_result_2 = df_result[result_columns_2]
    df_result_2_iterator = batch_iterator(df_result_2)
    eval_config_1 = dict(threshold=0.1)
    eval_config_2 = dict(threshold=0.2)

    response = _upload_results(
        dataset_name,
        model_name,
        [(eval_config_1, df_result_1_iterator), (eval_config_2, df_result_2_iterator)],
    )
    assert response.n_inserted == 14
    assert response.n_updated == 0

    fetched_df_dp, df_results_by_eval = download_results(dataset_name, model_name)
    assert len(df_results_by_eval) == 2
    expected_df_dp = df_dp[3:10].reset_index(drop=True)
    assert_frame_equal(fetched_df_dp, expected_df_dp, dp_columns)
    df_results_by_eval = sorted(df_results_by_eval, key=lambda x: x[0].get("threshold"))
    fetched_eval_config_1, fetched_df_result_1 = df_results_by_eval[0]
    fetched_eval_config_2, fetched_df_result_2 = df_results_by_eval[1]
    assert fetched_eval_config_1 == eval_config_1
    assert fetched_eval_config_2 == eval_config_2
    expected_df_result_1 = df_result_1.drop(columns=[JOIN_COLUMN])[3:10].reset_index(drop=True)
    expected_df_result_2 = df_result_2.drop(columns=[JOIN_COLUMN])[3:10].reset_index(drop=True)
    result_columns_1.remove(JOIN_COLUMN)
    result_columns_2.remove(JOIN_COLUMN)
    assert_frame_equal(fetched_df_result_1, expected_df_result_1, result_columns_1)
    assert_frame_equal(fetched_df_result_2, expected_df_result_2, result_columns_2)


def test__upload_results__multiple_eval_configs__partial_uploading() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__upload_results__multiple_eval_configs__partial_uploading")
    model_name = with_test_prefix(f"{__file__}::test__upload_results__multiple_eval_configs__partial_uploading")
    df_dp = get_df_dp(10)
    dp_columns = [JOIN_COLUMN, "locator", "width", "height", "city"]
    upload_dataset(dataset_name, df_dp[dp_columns], id_fields=ID_FIELDS)

    df_result = get_df_result(10)
    input_result_columns_1 = [JOIN_COLUMN, "softmax_bitmap", "score"]
    input_result_columns_2 = [JOIN_COLUMN, "softmax_bitmap"]
    result_columns_1 = input_result_columns_1[1:]
    result_columns_2 = input_result_columns_2[1:]
    df_result_1_p1 = df_result[:5][input_result_columns_1]
    df_result_2_p1 = df_result[5:10][input_result_columns_2]
    eval_config_1 = dict(threshold=0.1)
    eval_config_2 = dict(threshold=0.2)

    response = _upload_results(
        dataset_name,
        model_name,
        [(eval_config_1, df_result_1_p1), (eval_config_2, df_result_2_p1)],
    )
    assert response.n_inserted == 10
    assert response.n_updated == 0

    expected_df_dp = df_dp.reset_index(drop=True)
    fetched_df_dp, df_results_by_eval = download_results(dataset_name, model_name)
    assert len(df_results_by_eval) == 2
    assert_frame_equal(fetched_df_dp, expected_df_dp, dp_columns)

    df_results_by_eval = sorted(df_results_by_eval, key=lambda x: x[0].get("threshold"))
    fetched_eval_config_1, fetched_df_result_1 = df_results_by_eval[0]
    fetched_eval_config_2, fetched_df_result_2 = df_results_by_eval[1]
    assert fetched_eval_config_1 == eval_config_1
    assert fetched_eval_config_2 == eval_config_2
    # verify the partial results with placeholder
    expected_df_result_1_partial = (
        df_result.drop(columns=[JOIN_COLUMN])[result_columns_1].reset_index(drop=True).astype("object")
    )
    expected_df_result_1_partial[5:10] = None
    expected_df_result_2_partial = (
        df_result.drop(columns=[JOIN_COLUMN])[result_columns_2].reset_index(drop=True).astype("object")
    )
    expected_df_result_2_partial[:5] = None
    assert_frame_equal(fetched_df_result_1, expected_df_result_1_partial, result_columns_1)
    assert_frame_equal(fetched_df_result_2, expected_df_result_2_partial, result_columns_2)

    # insert the missing results, they will have full results
    df_result_1_p2 = df_result[5:10][input_result_columns_1]
    df_result_2_p2 = df_result[:5][input_result_columns_2]
    response = _upload_results(
        dataset_name,
        model_name,
        [(eval_config_1, df_result_1_p2), (eval_config_2, df_result_2_p2)],
    )
    assert response.n_inserted == 10
    assert response.n_updated == 0

    fetched_df_dp, df_results_by_eval = download_results(dataset_name, model_name)
    assert len(df_results_by_eval) == 2
    assert_frame_equal(fetched_df_dp, expected_df_dp, dp_columns)

    df_results_by_eval = sorted(df_results_by_eval, key=lambda x: x[0].get("threshold"))
    fetched_eval_config_1, fetched_df_result_1 = df_results_by_eval[0]
    fetched_eval_config_2, fetched_df_result_2 = df_results_by_eval[1]
    assert fetched_eval_config_1 == eval_config_1
    assert fetched_eval_config_2 == eval_config_2
    expected_df_result_1 = df_result.drop(columns=[JOIN_COLUMN])[result_columns_1].reset_index(drop=True)
    expected_df_result_2 = df_result.drop(columns=[JOIN_COLUMN])[result_columns_2].reset_index(drop=True)
    assert_frame_equal(fetched_df_result_1, expected_df_result_1, result_columns_1)
    assert_frame_equal(fetched_df_result_2, expected_df_result_2, result_columns_2)


def test__upload_results__multiple_eval_configs__duplicate() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__upload_results__multiple_eval_configs__duplicate")
    model_name = with_test_prefix(f"{__file__}::test__upload_results__multiple_eval_configs__duplicate")
    df_dp = get_df_dp()
    dp_columns = [JOIN_COLUMN, "locator", "width", "height", "city"]
    upload_dataset(dataset_name, df_dp[3:10][dp_columns], id_fields=ID_FIELDS)

    df_result = get_df_result()
    result_columns_1 = [JOIN_COLUMN, "softmax_bitmap", "score"]
    result_columns_2 = [JOIN_COLUMN, "softmax_bitmap"]
    df_result_1 = df_result[result_columns_1]
    df_result_2 = df_result[result_columns_2]
    eval_config = dict(threshold=0.1)

    with pytest.raises(IncorrectUsageError) as exc_info:
        _upload_results(
            dataset_name,
            model_name,
            [(eval_config, df_result_1), (eval_config, df_result_2)],
        )

    exc_info_value = str(exc_info.value)
    assert "duplicate eval configs are invalid" in exc_info_value


def test__upload_results__missing_result() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__upload_results__missing_result")
    model_name = with_test_prefix(f"{__file__}::test__upload_results__missing_result")
    df_dp = get_df_dp()
    dp_columns = [JOIN_COLUMN, "locator", "width", "height", "city"]
    upload_dataset(dataset_name, df_dp[3:10][dp_columns], id_fields=ID_FIELDS)

    df_result = get_df_result()
    result_columns = ["softmax_bitmap", "score"]

    response = _upload_results(
        dataset_name,
        model_name,
        df_result,
    )
    assert response.n_inserted == 7
    assert response.n_updated == 0

    fetched_df_dp, df_results_by_eval = download_results(dataset_name, model_name)
    eval_cfg, fetched_df_result = df_results_by_eval[0]
    assert len(df_results_by_eval) == 1
    assert eval_cfg is None
    expected_df_dp = df_dp[3:10].reset_index(drop=True)
    expected_df_result = df_result.drop(columns=[JOIN_COLUMN])[3:10].reset_index(drop=True)
    assert_frame_equal(fetched_df_dp, expected_df_dp, dp_columns)
    assert_frame_equal(fetched_df_result, expected_df_result, result_columns)

    # add 3 new datapoints, then we should have missing results in the db records
    upload_dataset(dataset_name, df_dp[:10][dp_columns], id_fields=ID_FIELDS)
    fetched_df_dp, df_results_by_eval = download_results(dataset_name, model_name)
    eval_cfg, fetched_df_result = df_results_by_eval[0]
    assert len(df_results_by_eval) == 1
    assert eval_cfg is None
    expected_df_dp = pd.concat([df_dp[3:10], df_dp[:3]]).sort_values(JOIN_COLUMN).reset_index(drop=True)
    expected_df_result = pd.concat(
        [
            df_result.drop(columns=[JOIN_COLUMN])[3:10].astype("object"),
            pd.DataFrame({"softmax_bitmap": [None] * 3, "score": [None] * 3}),
        ],
        axis=0,
    ).reset_index(drop=True)
    fetched_df_dp = fetched_df_dp.sort_values(JOIN_COLUMN)
    expected_df_dp = expected_df_dp.sort_values(JOIN_COLUMN)
    fetched_df_dp.reset_index(drop=True, inplace=True)
    expected_df_dp.reset_index(drop=True, inplace=True)
    assert_frame_equal(fetched_df_dp, expected_df_dp, dp_columns)
    assert_frame_equal(fetched_df_result, expected_df_result, result_columns)


def test__upload_results__upload_none() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__upload_results__upload_none")
    model_name = with_test_prefix(f"{__file__}::test__upload_results__upload_none")
    df_dp = get_df_dp()
    dp_columns = [JOIN_COLUMN, "locator", "width", "height", "city"]
    upload_dataset(dataset_name, df_dp[dp_columns], id_fields=ID_FIELDS)

    df_result = get_df_result(10)
    result_columns = ["softmax_bitmap", "score"]

    response = _upload_results(
        dataset_name,
        model_name,
        df_result,
    )
    assert response.n_inserted == 10
    assert response.n_updated == 0

    fetched_df_dp, df_results_by_eval = download_results(dataset_name, model_name)
    eval_cfg, fetched_df_result = df_results_by_eval[0]
    expected_df_dp = df_dp.reset_index(drop=True)
    expected_df_result = pd.concat(
        [
            df_result.drop(columns=JOIN_COLUMN).astype("object"),
            pd.DataFrame({"softmax_bitmap": [None] * 10, "score": [None] * 10}),
        ],
        axis=0,
    ).reset_index(
        drop=True,
    )
    assert len(df_results_by_eval) == 1
    assert eval_cfg is None
    assert_frame_equal(fetched_df_dp, expected_df_dp, dp_columns)
    assert_frame_equal(fetched_df_result, expected_df_result, result_columns)


def test__upload_results__thresholded() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__upload_results__thresholded")
    model_name = with_test_prefix(f"{__file__}::test__upload_results__thresholded")
    df_dp = get_df_dp()
    dp_columns = [JOIN_COLUMN, "locator", "width", "height", "city"]
    upload_dataset(dataset_name, df_dp[3:10][dp_columns], id_fields=ID_FIELDS)

    records = [
        dict(
            user_dp_id=i,
            softmax_bitmap=fake_locator(i, "softmax_bitmap"),
            score=i * 0.1,
            bev=[dict(threshold=(j + 1) * 0.1, label="cat", foo=i + j) for j in range(3)],
        )
        for i in range(20)
    ]
    df_result = pd.DataFrame(records)
    result_columns = ["softmax_bitmap", "score", "bev"]
    response = _upload_results(
        dataset_name,
        model_name,
        df_result,
        thresholded_fields=["bev"],
    )
    assert response.n_inserted == 7
    assert response.n_updated == 0

    fetched_df_dp, df_results_by_eval = download_results(dataset_name, model_name)
    eval_cfg, fetched_df_result = df_results_by_eval[0]
    assert len(df_results_by_eval) == 1
    assert eval_cfg is None
    expected_df_dp = df_dp[3:10].reset_index(drop=True)
    expected_df_result = df_result.drop(columns=[JOIN_COLUMN])[3:10].reset_index(drop=True)
    assert_frame_equal(fetched_df_dp, expected_df_dp, dp_columns)
    assert_frame_equal(fetched_df_result, expected_df_result, result_columns)


def test__download_results__not_exist() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__download_results__not_exist")
    model_name = with_test_prefix(f"{__file__}::test__download_results__not_exist")
    with pytest.raises(NotFoundError) as exc_info:
        download_results(dataset_name, model_name)
    exc_info_value = str(exc_info.value)
    assert "no such model" in exc_info_value


def test__download_results__reset_dataset() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__download_results__reset_dataset")
    model_name = with_test_prefix(f"{__file__}::test__download_results__reset_dataset")

    df_dp = get_df_dp()
    dp_columns = [JOIN_COLUMN, "locator", "width", "height", "city"]
    upload_dataset(dataset_name, df_dp[dp_columns], id_fields=ID_FIELDS)

    df_result = get_df_result(10)
    eval_config = dict(threshold=0.422)

    response = _upload_results(
        dataset_name,
        model_name,
        [(eval_config, df_result)],
    )
    assert response.n_inserted == 10
    assert response.n_updated == 0

    fetched_df_dp, df_results_by_eval = download_results(dataset_name, model_name)
    eval_cfg, fetched_df_result = df_results_by_eval[0]
    assert not fetched_df_dp.empty
    assert not fetched_df_result.empty
    assert len(df_results_by_eval) == 1
    assert eval_cfg == eval_config

    # reset dataset by updating id_fields, no results are kept
    upload_dataset(dataset_name, df_dp[dp_columns], id_fields=[*ID_FIELDS, "locator"])

    fetched_df_dp, df_results_by_eval = download_results(dataset_name, model_name)
    assert fetched_df_dp.empty
    assert len(df_results_by_eval) == 0


def test__download_results__preserve_none() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__download_results__preserve_none")
    model_name = with_test_prefix(f"{__file__}::test__download_results__preserve_none")

    data = [{"a": None, "id": 1}, {"a": float("inf"), "id": 2}, {"a": np.nan, "id": 3}, {"a": 42, "id": 4}]
    df_dp = pd.DataFrame.from_dict(data, dtype=object)
    assert df_dp["a"][0] is None
    assert np.isinf(df_dp["a"][1])
    assert np.isnan(df_dp["a"][2])
    upload_dataset(dataset_name, df_dp, id_fields=["id"])

    response = _upload_results(dataset_name, model_name, df_dp)
    assert response.n_inserted == 4
    assert response.n_updated == 0

    fetched_df_dp, df_results_by_eval = download_results(dataset_name, model_name)
    eval_cfg, fetched_df_result = df_results_by_eval[0]
    assert not fetched_df_dp.empty
    assert not fetched_df_result.empty
    assert len(df_results_by_eval) == 1
    assert eval_cfg is None

    assert_frame_equal(fetched_df_dp, df_dp)
    assert_frame_equal(fetched_df_result, df_dp[["a"]])
    assert fetched_df_dp["a"][0] is None
    assert np.isinf(fetched_df_dp["a"][1])
    assert np.isnan(fetched_df_dp["a"][2])
