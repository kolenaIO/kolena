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
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kolena._experimental.quality_standard import download_quality_standard_result
from kolena.dataset import upload_dataset
from kolena.dataset.evaluation import _upload_results
from tests.integration._experimental.helper import create_quality_standard
from tests.integration.helper import fake_locator
from tests.integration.helper import with_test_prefix

N_DATAPOINTS = 20
ID_FIELDS = ["locator"]


@pytest.fixture
def datapoint_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(
                locator=fake_locator(i, "datapoints"),
                width=i + 500,
                height=i + 400,
                city=["new york", "waterloo"][i % 2],
            )
            for i in range(N_DATAPOINTS)
        ],
    )


@pytest.fixture
def result_df() -> pd.DataFrame:
    return pd.DataFrame([dict(locator=fake_locator(i, "datapoints"), score=i * 0.1) for i in range(N_DATAPOINTS)])


def test__download_quality_standard_result(datapoint_df: pd.DataFrame, result_df: pd.DataFrame) -> None:
    dataset_name = with_test_prefix("test__download_quality_standard_result__dataset")
    model_name = with_test_prefix("test__download_quality_standard_result__model")
    upload_dataset(dataset_name, datapoint_df, id_fields=ID_FIELDS)
    _upload_results(
        dataset_name,
        model_name,
        result_df,
    )

    metric_group_name = "test group"
    metric_name = "Min Score"
    test_case_name = "city"
    quality_standard = dict(
        name=with_test_prefix("test__download_quality_standard_result__quality_standard"),
        stratifications=[
            dict(
                name=test_case_name,
                stratify_fields=[dict(source="datapoint", field="city", values=["new york", "waterloo"])],
                test_cases=[
                    dict(name="new york", stratification=[dict(value="new york")]),
                    dict(name="waterloo", stratification=[dict(value="waterloo")]),
                ],
            ),
        ],
        metric_groups=[
            dict(
                name=metric_group_name,
                metrics=[dict(label=metric_name, source="result", aggregator="min", params=dict(key="score"))],
            ),
        ],
        version="1.0",
    )
    create_quality_standard(dataset_name, quality_standard)

    quality_standard_df = download_quality_standard_result(dataset_name, [model_name])
    assert_frame_equal(
        quality_standard_df,
        download_quality_standard_result(dataset_name, [model_name], [metric_group_name]),
    )

    df_columns: pd.MultiIndex = quality_standard_df.columns
    assert df_columns.names == ["model", "eval_config", "metric_group", "metric"]
    assert df_columns.levels == [[model_name], ["null"], [metric_group_name], [metric_name]]

    df_index: pd.MultiIndex = quality_standard_df.index
    assert df_index.names == ["stratification", "test_case"]
    assert all(df_index.levels[0] == ["Dataset", test_case_name])
    assert all(df_index.levels[1] == ["new york", "waterloo"])

    assert quality_standard_df.loc[("Dataset", np.nan), (model_name, "null", metric_group_name, metric_name)] == 0.0
    assert quality_standard_df.loc[("city", "new york"), (model_name, "null", metric_group_name, metric_name)] == 0.0
    assert quality_standard_df.loc[("city", "waterloo"), (model_name, "null", metric_group_name, metric_name)] == 0.1
