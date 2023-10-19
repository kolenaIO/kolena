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
from typing import Optional

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic.dataclasses import dataclass

from kolena._experimental.dataset import fetch_evaluation_results
from kolena._experimental.dataset import fetch_inferences
from kolena._experimental.dataset import register_dataset
from kolena._experimental.dataset import test
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


def test__test() -> None:
    def get_df_inf() -> pd.DataFrame:
        record = [dict(softmax_bitmap=fake_locator(i, "softmax_bitmap")) for i in range(10)]
        columns = ["softmax_bitmap"]
        return pd.DataFrame(record, columns=columns)

    def get_df_metrics() -> pd.DataFrame:
        record = [dict(score=i) for i in range(10)]
        columns = ["score"]
        return pd.DataFrame(record, columns=columns)

    df_inf = get_df_inf()
    df_metrics = get_df_metrics()

    def infer_func(datapoints: pd.DataFrame) -> pd.DataFrame:
        return df_inf

    def eval_func(
        datapoints: pd.DataFrame,
        inferences: pd.DataFrame,
        eval_config: ThresholdConfiguration,
    ) -> pd.DataFrame:
        return df_metrics

    dataset_name = with_test_prefix(f"{__file__}::test__fetch_inferences__not_exist")
    model_name = with_test_prefix(f"{__file__}::test__fetch_inferences__not_exist")
    datapoints = [
        dict(
            locator=fake_locator(i, "datapoints"),
            width=i + 500,
            height=i + 400,
            city=random.choice(["new york", "waterloo"]),
        )
        for i in range(20)
    ]
    columns = ["locator", "width", "height", "city", "bboxes"]
    register_dataset(dataset_name, pd.DataFrame(datapoints[:10], columns=columns))

    test(
        dataset_name,
        model_name,
        infer=infer_func,
        eval=eval_func,
        eval_configs=[
            ThresholdConfiguration(
                threshold=0.3,
            ),
            ThresholdConfiguration(
                threshold=0.6,
            ),
        ],
    )
    df_by_eval = fetch_evaluation_results(dataset_name, model_name)
    assert len(df_by_eval) == 2
    # TODO: figure out the expected
    expected = pd.DataFrame(datapoints[:10], columns=columns)
    assert_frame_equal(df_by_eval[0], expected)


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
