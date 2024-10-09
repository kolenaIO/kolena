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
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kolena._experimental import copy_quality_standards_from_dataset
from kolena._experimental import download_quality_standard_result
from kolena.dataset import upload_dataset
from kolena.dataset.evaluation import _upload_results
from kolena.dataset.evaluation import EvalConfig
from kolena.errors import IncorrectUsageError
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


def test__download_quality_standard_result_with_performance_delta(
    datapoints: pd.DataFrame,
    results: List[Tuple[EvalConfig, pd.DataFrame]],
) -> None:
    dataset_name = with_test_prefix("test__download_quality_standard_result_with_performance_delta")
    model_name = with_test_prefix("test__test__download_quality_standard_result_with_performance_delta_model")
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
    metric_value_label = "metric_value"
    performance_delta_label = "performance_delta"
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
                    dict(
                        label="Max Score",
                        source="result",
                        aggregator="max",
                        params=dict(key="score"),
                        highlight=dict(higherIsBetter=True),
                    ),
                    dict(
                        label="Min Score",
                        source="result",
                        aggregator="min",
                        params=dict(key="score"),
                        highlight=dict(higherIsBetter=False),
                    ),
                ],
            ),
        ],
        version="1.0",
    )
    create_quality_standard(dataset_name, quality_standard)

    quality_standard_df = download_quality_standard_result(
        dataset_name,
        [model_name],
        confidence_level=0.95,
        reference_eval_config="null",
    )

    df_columns: pd.MultiIndex = quality_standard_df.columns
    assert df_columns.names == ["model", "eval_config", "metric_group", "metric", "type"]
    assert all(df_columns.levels[0] == [model_name])
    assert all(df_columns.levels[1] == [json.dumps(eval_config) for eval_config in eval_configs])
    assert all(df_columns.levels[2] == [metric_group_name])
    assert all(df_columns.levels[3] == [max_metric_label, min_metric_label])
    assert all(df_columns.levels[4] == [metric_value_label, performance_delta_label])

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
            quality_standard_df.loc[
                ("Dataset", np.nan),
                (
                    model_name,
                    json_config,
                    metric_group_name,
                    max_metric_label,
                    metric_value_label,
                ),
            ]
            == dataset_maximum
        )
        assert (
            quality_standard_df.loc[
                ("city", "new york"),
                (model_name, json_config, metric_group_name, max_metric_label, metric_value_label),
            ]
            == newyork_maximum
        )
        assert (
            quality_standard_df.loc[
                ("city", "waterloo"),
                (model_name, json_config, metric_group_name, max_metric_label, metric_value_label),
            ]
            == waterloo_maximum
        )

    performance_delta_min_metric = list(
        quality_standard_df[
            (
                model_name,
                json.dumps(eval_configs[1]),
                metric_group_name,
                min_metric_label,
                performance_delta_label,
            )
        ],
    )
    assert performance_delta_min_metric == ["similar", "similar", "regressed"]
    performance_delta_max_metric = list(
        quality_standard_df[
            (
                model_name,
                json.dumps(eval_configs[1]),
                metric_group_name,
                max_metric_label,
                performance_delta_label,
            )
        ],
    )
    assert performance_delta_max_metric == ["improved", "improved", "improved"]
    assert (
        list(
            quality_standard_df[
                (
                    model_name,
                    json.dumps(eval_configs[0]),
                    metric_group_name,
                    min_metric_label,
                    performance_delta_label,
                )
            ],
        )
        == list(
            quality_standard_df[
                (
                    model_name,
                    json.dumps(eval_configs[0]),
                    metric_group_name,
                    max_metric_label,
                    performance_delta_label,
                )
            ],
        )
        == ["similar", "similar", "similar"]
    )


def test__copy_quality_standards_from_dataset__dataset_same_as_source() -> None:
    dataset_name = with_test_prefix("test__copy_quality_standards_from_dataset__dataset_same_as_source")
    source_dataset_name = dataset_name
    with pytest.raises(IncorrectUsageError) as exc_info:
        copy_quality_standards_from_dataset(dataset_name, source_dataset_name)
    exc_info_value = str(exc_info.value)
    assert "source dataset and target dataset are the same" in exc_info_value


def _assert_metric_groups_equal(metric_groups_1: List[Dict[str, Any]], metric_groups_2: List[Dict[str, Any]]) -> None:
    assert len(metric_groups_1) == len(metric_groups_2)
    for metric_group_1, metric_group_2 in zip(metric_groups_1, metric_groups_2):
        assert metric_group_1["name"] == metric_group_2["name"]
        assert len(metric_group_1["metrics"]) == len(metric_group_2["metrics"])
        for metric_1, metric_2 in zip(metric_group_1["metrics"], metric_group_2["metrics"]):
            assert metric_1["label"] == metric_2["label"]


def _assert_test_cases_equal(test_cases_list_1: List[Dict[str, Any]], test_cases_list_2: List[Dict[str, Any]]) -> None:
    assert len(test_cases_list_1) == len(test_cases_list_2)
    for test_cases_1, test_cases_2 in zip(test_cases_list_1, test_cases_list_2):
        assert test_cases_1["name"] == test_cases_2["name"]
        assert len(test_cases_1["test_cases"]) == len(test_cases_2["test_cases"])
        for metric_1, metric_2 in zip(test_cases_1["test_cases"], test_cases_2["test_cases"]):
            assert metric_1["name"] == metric_2["name"]


def test__copy_quality_standards_from_dataset(datapoints: pd.DataFrame) -> None:
    source_dataset_name = with_test_prefix("test__copy_quality_standards_from_dataset__source_dataset")
    dataset_name = with_test_prefix("test__copy_quality_standards_from_dataset__dataset")

    upload_dataset(source_dataset_name, datapoints, id_fields=ID_FIELDS)
    upload_dataset(dataset_name, datapoints, id_fields=ID_FIELDS)

    quality_standards = dict(
        name=with_test_prefix("test__copy_quality_standards_from_dataset__qs"),
        stratifications=[
            dict(
                name=with_test_prefix("test__copy_quality_standards_from_dataset__test-case"),
                stratify_fields=[dict(source="datapoint", field="city", values=["new york", "waterloo"])],
                test_cases=[
                    dict(name="new york", stratification=[dict(value="new york")]),
                    dict(name="waterloo", stratification=[dict(value="waterloo")]),
                ],
            ),
        ],
        metric_groups=[
            dict(
                name=with_test_prefix("test__copy_quality_standards_from_dataset__metric_group"),
                metrics=[
                    dict(label="Max Score", source="result", aggregator="max", params=dict(key="score")),
                    dict(label="Min Score", source="result", aggregator="min", params=dict(key="score")),
                ],
            ),
        ],
        version="1.0",
    )
    create_quality_standard(source_dataset_name, quality_standards)

    # by default, should copy both metric groups and test cases
    metric_groups, test_cases = copy_quality_standards_from_dataset(dataset_name, source_dataset_name)
    _assert_metric_groups_equal(quality_standards["metric_groups"], metric_groups)
    _assert_test_cases_equal(quality_standards["stratifications"], test_cases)

    # exclude metric groups
    metric_groups, test_cases = copy_quality_standards_from_dataset(
        dataset_name,
        source_dataset_name,
        include_metric_groups=False,
    )
    assert metric_groups == []
    _assert_test_cases_equal(quality_standards["stratifications"], test_cases)

    # exclude test cases
    metric_groups, test_cases = copy_quality_standards_from_dataset(
        dataset_name,
        source_dataset_name,
        include_test_cases=False,
    )
    _assert_metric_groups_equal(quality_standards["metric_groups"], metric_groups)
    assert test_cases == []

    # cannot exclude both test cases and metric groups
    with pytest.raises(IncorrectUsageError) as exc_info:
        copy_quality_standards_from_dataset(
            dataset_name,
            source_dataset_name,
            include_metric_groups=False,
            include_test_cases=False,
        )
    exc_info_value = str(exc_info.value)
    assert "should include at least one of metric groups or test cases" in exc_info_value
