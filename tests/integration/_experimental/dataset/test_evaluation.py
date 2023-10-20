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

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic.dataclasses import dataclass

from kolena._experimental.dataset import fetch_evaluation_results
from kolena._experimental.dataset import fetch_inferences
from kolena._experimental.dataset import register_dataset
from kolena._experimental.dataset import test
from kolena._experimental.dataset._evaluation import EVAL_FUNC_TYPE
from kolena._experimental.dataset._evaluation import INFER_FUNC_TYPE
from kolena.errors import NotFoundError
from kolena.workflow import EvaluatorConfiguration
from tests.integration.helper import fake_locator
from tests.integration.helper import with_test_prefix


@dataclass(frozen=True)
class ThresholdConfiguration(EvaluatorConfiguration):
    threshold: Optional[float] = None

    def display_name(self) -> str:
        if self.threshold is not None:
            return f"Confidence Above Threshold (threshold={self.threshold})"
        return "Max Confidence"


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


def get_df_metrics(n: int = 20) -> pd.DataFrame:
    records = [dict(user_dp_id=i, score=i) for i in range(n)]
    return pd.DataFrame(records)


def get_infer_func(
    df_inf: pd.DataFrame,
    columns: List[str],
    id_col: str = "user_dp_id",
    how: str = "left",
) -> INFER_FUNC_TYPE:
    def infer_func(datapoints: pd.DataFrame) -> pd.DataFrame:
        return datapoints.set_index(id_col).join(df_inf.set_index(id_col), how=how).reset_index()[columns]

    return infer_func


def get_eval_func(
    df_metrics: pd.DataFrame,
    columns: List[str],
    id_col: str = "user_dp_id",
    how: str = "left",
) -> EVAL_FUNC_TYPE:
    def eval_func(
        datapoints: pd.DataFrame,
        inferences: pd.DataFrame,
        eval_config: ThresholdConfiguration,
    ) -> pd.DataFrame:
        return datapoints.set_index(id_col).join(df_metrics.set_index(id_col), how=how).reset_index()[columns]

    return eval_func


def test__test() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__test")
    model_name = with_test_prefix(f"{__file__}::test__test")
    df_dp = get_df_dp()
    dp_columns = ["user_dp_id", "locator", "width", "height", "city"]
    register_dataset(dataset_name, df_dp[:10][dp_columns])

    df_inf = get_df_inf()
    df_metrics = get_df_metrics()
    inf_columns = ["softmax_bitmap"]
    metrics_columns = ["score"]

    eval_configs = [
        ThresholdConfiguration(
            threshold=0.3,
        ),
        ThresholdConfiguration(
            threshold=0.6,
        ),
    ]
    test(
        dataset_name,
        model_name,
        infer=get_infer_func(df_inf, inf_columns),
        eval=get_eval_func(df_metrics, metrics_columns),
        eval_configs=eval_configs,
    )

    df_by_eval = fetch_evaluation_results(dataset_name, model_name)
    assert len(df_by_eval) == 2
    assert sorted([cfg for cfg, *_ in df_by_eval], key=lambda x: x.get("threshold")) == sorted(
        [dict(cfg._to_dict()) for cfg in eval_configs],
        key=lambda x: x.get("threshold"),
    )
    expected_df_dp = df_dp[:10][dp_columns]
    expected_df_inf = df_inf[:10][inf_columns]
    expected_df_metrics = df_metrics[:10][metrics_columns]
    for df_eval in df_by_eval:
        assert_frame_equal(df_eval[1][dp_columns], expected_df_dp)
        assert_frame_equal(df_eval[2], expected_df_inf)
        assert_frame_equal(df_eval[3], expected_df_metrics)


def test__fetch_inferences__not_exist() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__fetch_inferences__not_exist")
    model_name = with_test_prefix(f"{__file__}::test__fetch_inferences__not_exist")
    df_datapoints, df_inferences = fetch_inferences(dataset_name, model_name)
    assert df_datapoints.empty
    assert df_inferences.empty


def test__fetch_evaluation_results_not_exist() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__fetch_evaluation_results_not_exist")
    model_name = with_test_prefix(f"{__file__}::test__fetch_evaluation_results_not_exist")
    with pytest.raises(NotFoundError):
        fetch_evaluation_results(dataset_name, model_name)
