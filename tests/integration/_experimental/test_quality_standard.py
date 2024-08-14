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
import json
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kolena._experimental import download_quality_standard_result
from kolena.dataset import upload_dataset
from kolena.dataset.evaluation import _upload_results
from kolena.dataset.evaluation import EvalConfig
from tests.integration._experimental.helper import create_quality_standard
from tests.integration.helper import fake_locator
from tests.integration.helper import with_test_prefix

N_DATAPOINTS = 10
ID_FIELDS = ["locator"]


@pytest.fixture
def datapoints() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(
                locator=fake_locator(i, "datapoints"),
                city=["new york", "waterloo"][i % 2],
            )
            for i in range(N_DATAPOINTS)
        ],
    )


@pytest.fixture
def results() -> List[Tuple[EvalConfig, pd.DataFrame]]:
    return [
        (None, pd.DataFrame([dict(locator=fake_locator(i, "datapoints"), score=i * 0.1) for i in range(N_DATAPOINTS)])),
        (
            dict(double=True),
            pd.DataFrame([dict(locator=fake_locator(i, "datapoints"), score=i * 0.2) for i in range(N_DATAPOINTS)]),
        ),
    ]


def test__download_quality_standard_result(
    datapoints: pd.DataFrame,
    results: List[Tuple[EvalConfig, pd.DataFrame]],
) -> None:
    dataset_name = with_test_prefix("test__download_quality_standard_result__dataset")
    model_name = with_test_prefix("test__download_quality_standard_result__model")
    eval_configs = [eval_config for eval_config, _ in results]
    upload_dataset(dataset_name, datapoints, id_fields=ID_FIELDS)
    _upload_results(
        dataset_name,
        model_name,
        results,
    )

    test_case_name = "city"
    metric_group_name = "test group"
    max_metric_label = "Max Score"
    min_metric_label = "Min Score"
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
                metrics=[
                    dict(label="Max Score", source="result", aggregator="max", params=dict(key="score")),
                    dict(label="Min Score", source="result", aggregator="min", params=dict(key="score")),
                ],
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
    assert all(df_columns.levels[0] == [model_name])
    assert all(df_columns.levels[1] == [json.dumps(eval_config) for eval_config in eval_configs])
    assert all(df_columns.levels[2] == [metric_group_name])
    assert all(df_columns.levels[3] == [max_metric_label, min_metric_label])

    df_index: pd.MultiIndex = quality_standard_df.index
    assert df_index.names == ["stratification", "test_case"]
    assert all(df_index.levels[0] == ["Dataset", test_case_name])
    assert all(df_index.levels[1] == ["new york", "waterloo"])

    for eval_config in eval_configs:
        json_config = json.dumps(eval_config)

        newyork_maximum = 0.8 * 2 if eval_config else 0.8
        waterloo_maximum = 0.9 * 2 if eval_config else 0.9
        dataset_maximum = max(newyork_maximum, waterloo_maximum)

        assert (
            quality_standard_df.loc[("Dataset", np.nan), (model_name, json_config, metric_group_name, max_metric_label)]
            == dataset_maximum
        )
        assert (
            quality_standard_df.loc[
                ("city", "new york"),
                (model_name, json_config, metric_group_name, max_metric_label),
            ]
            == newyork_maximum
        )
        assert (
            quality_standard_df.loc[
                ("city", "waterloo"),
                (model_name, json_config, metric_group_name, max_metric_label),
            ]
            == waterloo_maximum
        )

        newyork_minimum = 0.0
        waterloo_minimum = 0.1 * 2 if eval_config else 0.1
        dataset_minimum = min(newyork_minimum, waterloo_minimum)
        assert (
            quality_standard_df.loc[("Dataset", np.nan), (model_name, json_config, metric_group_name, min_metric_label)]
            == dataset_minimum
        )
        assert (
            quality_standard_df.loc[
                ("city", "new york"),
                (model_name, json_config, metric_group_name, min_metric_label),
            ]
            == newyork_minimum
        )
        assert (
            quality_standard_df.loc[
                ("city", "waterloo"),
                (model_name, json_config, metric_group_name, min_metric_label),
            ]
            == waterloo_minimum
        )


def test__download_quality_standard_result__union(
    datapoints: pd.DataFrame,
    results: List[Tuple[EvalConfig, pd.DataFrame]],
) -> None:
    dataset_name = with_test_prefix("test__download_quality_standard_result__union")
    model_name = with_test_prefix("test__download_quality_standard_result__union_model")
    model_name2 = with_test_prefix("test__download_quality_standard_result__union_model_2")
    eval_config, model_1_results = results[0]
    upload_dataset(dataset_name, datapoints, id_fields=ID_FIELDS)
    _upload_results(
        dataset_name,
        model_name,
        model_1_results,
    )
    model_2_results = model_1_results[2:]
    _upload_results(
        dataset_name,
        model_name2,
        model_2_results,
    )

    test_case_name = "city"
    metric_group_name = "test group"
    min_metric_label = "Min Score"
    quality_standard = dict(
        name=with_test_prefix("test__download_quality_standard_result__union"),
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
                metrics=[
                    dict(label="Min Score", source="result", aggregator="min", params=dict(key="score")),
                ],
            ),
        ],
        version="1.0",
    )
    create_quality_standard(dataset_name, quality_standard)

    quality_standard_df_intersection = download_quality_standard_result(dataset_name, [model_name, model_name2])
    assert_frame_equal(
        quality_standard_df_intersection,
        download_quality_standard_result(dataset_name, [model_name, model_name2], [metric_group_name]),
    )
    quality_standard_df_union = download_quality_standard_result(
        dataset_name,
        [model_name, model_name2],
        intersect_results=False,
    )

    json_config = json.dumps(eval_config)

    newyork_minimum = 0.2
    waterloo_minimum = 0.3
    dataset_minimum = min(newyork_minimum, waterloo_minimum)
    assert (
        quality_standard_df_intersection.loc[
            ("Dataset", np.nan),
            (model_name, json_config, metric_group_name, min_metric_label),
        ]
        == dataset_minimum
    )
    assert (
        quality_standard_df_intersection.loc[
            ("city", "new york"),
            (model_name, json_config, metric_group_name, min_metric_label),
        ]
        == newyork_minimum
    )
    assert (
        quality_standard_df_intersection.loc[
            ("city", "waterloo"),
            (model_name, json_config, metric_group_name, min_metric_label),
        ]
        == waterloo_minimum
    )

    newyork_minimum = 0.0
    waterloo_minimum = 0.1
    dataset_minimum = min(newyork_minimum, waterloo_minimum)
    assert (
        quality_standard_df_union.loc[
            ("Dataset", np.nan),
            (model_name, json_config, metric_group_name, min_metric_label),
        ]
        == dataset_minimum
    )
    assert (
        quality_standard_df_union.loc[
            ("city", "new york"),
            (model_name, json_config, metric_group_name, min_metric_label),
        ]
        == newyork_minimum
    )
    assert (
        quality_standard_df_union.loc[
            ("city", "waterloo"),
            (model_name, json_config, metric_group_name, min_metric_label),
        ]
        == waterloo_minimum
    )
