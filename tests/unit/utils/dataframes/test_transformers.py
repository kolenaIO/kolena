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
import math
from pathlib import Path
from typing import Any
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from kolena._utils.dataframes.transformers import df_apply
from kolena._utils.dataframes.transformers import drop_invalid_ground_truth
from kolena._utils.dataframes.transformers import drop_invalid_metadata
from kolena._utils.dataframes.transformers import drop_unnamed
from kolena._utils.dataframes.transformers import json_normalize
from kolena._utils.dataframes.transformers import parse_cols


@pytest.fixture
def nested_data() -> Dict:
    data = [
        {"id": 1, "name": {"first": "Coleen", "last": "Volk"}},
        {"name": {"given": "Mark", "family": "Regner"}},
        {"id": 3, "name": "Faye Raker"},
    ]
    return data


def _try_convert_to_num(s: str) -> Any:
    if s == "":
        return None

    try:
        return pd.to_numeric(s)
    except Exception:
        return s


def test__df_apply() -> None:
    df = pd.DataFrame({"A": ["", "2", "3"]})

    assert df["A"][0] == ""
    df_post = df["A"].apply(_try_convert_to_num)
    assert math.isnan(df_post[0])
    # verify convert_dtype is False
    df_post = df_apply(df["A"], _try_convert_to_num)
    assert df_post[0] is None


def test__json_normalize(nested_data: Dict):
    max_level = 0
    normalized_data = [
        {"id": 1, "name": {"first": "Coleen", "last": "Volk"}},
        {"name": {"given": "Mark", "family": "Regner"}, "id": None},
        {"id": 3, "name": "Faye Raker"},
    ]
    df_expected = pd.DataFrame.from_dict(normalized_data, dtype=object)
    df_post = json_normalize(nested_data, max_level=max_level)
    # assert the None is used for missing value
    assert df_post["id"][1] is None
    pd.testing.assert_frame_equal(df_post, df_expected)

    max_level = 1
    normalized_data = [
        {"id": 1, "name.first": "Coleen", "name.last": "Volk", "name": None, "name.family": None, "name.given": None},
        {
            "name.given": "Mark",
            "name.family": "Regner",
            "id": None,
            "name": None,
            "name.last": None,
            "name.first": None,
        },
        {"id": 3, "name": "Faye Raker", "name.last": None, "name.family": None, "name.first": None, "name.given": None},
    ]
    df_expected = pd.DataFrame.from_dict(normalized_data, dtype=object)
    df_post = json_normalize(nested_data, max_level=max_level)
    # assert the None is used for missing value
    assert df_post["name"][0] is None
    assert df_post["name.given"][0] is None
    assert df_post["id"][1] is None
    # ignore the order of columns by check_like
    pd.testing.assert_frame_equal(df_post, df_expected, check_like=True)


DATA_DIR = Path(__file__).parent / "data"


def _get_csv_data(file_name: str, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(str(DATA_DIR / file_name), **kwargs)


def test__parse_cols() -> None:
    file_name = "parse_json.csv"
    df = _get_csv_data(file_name)
    cols = ["A", "B", "C", "D", "E"]
    assert df.columns.values.tolist() == cols
    assert df.iloc[0]["A"] == 1
    assert df.iloc[0]["B"] == "[];print('hack4')"
    assert df.iloc[0]["C"] == "{'foo': 7}"
    assert df.iloc[0]["D"] == '{"foo": 7}'
    assert math.isnan(df.iloc[0]["E"])

    new_df = parse_cols(df)
    assert new_df.iloc[0]["A"] == 1
    assert new_df.iloc[0]["B"] == "[];print('hack4')"
    assert new_df.iloc[0]["C"] == "{'foo': 7}"
    assert new_df.iloc[0]["D"] == {"foo": 7}
    assert math.isnan(new_df.iloc[0]["E"])

    assert len(new_df) == len(df)


def test__parse_cols__convert_ndarray() -> None:
    df = pd.DataFrame(dict(ndarray=[np.arange(3)]))
    assert isinstance(df.iloc[0]["ndarray"], np.ndarray)

    new_df = parse_cols(df)
    assert isinstance(new_df.iloc[0]["ndarray"], list)


def test__parse_cols__handle_na() -> None:
    file_name = "nan.csv"
    df = _get_csv_data(file_name)
    cols = ["A", "B", "C", "D"]
    assert df.columns.values.tolist() == cols
    assert df.iloc[0]["A"] == 1
    assert pd.isnull(df.iloc[0]["B"])
    assert df.iloc[0]["C"] == '["a","b","c"]'
    assert df.iloc[0]["D"] == False  # noqa: E712

    new_df = parse_cols(df)
    assert new_df.iloc[0]["A"] == 1
    assert new_df.iloc[0]["B"] is None
    assert new_df.iloc[0]["C"] == ["a", "b", "c"]
    assert new_df.iloc[0]["D"] == False  # noqa: E712
    assert len(new_df) == len(df)


def test__parse_cols__heterogeneous() -> None:
    file_name = "heterogeneous.csv"
    df = _get_csv_data(file_name, keep_default_na=False)
    cols = ["a", "b"]
    assert df.columns.values.tolist() == cols
    assert df.iloc[0]["a"] == ""
    assert df.iloc[1]["a"] == "3"
    assert df.iloc[2]["a"] == "4"
    assert df.iloc[3]["a"] == "6e2"
    assert df.iloc[4]["a"] == "62884541901610273646114"
    assert df.iloc[0]["b"] == "hi"
    assert df.iloc[1]["b"] == "  "
    assert df.iloc[2]["b"] == ""
    assert df.iloc[3]["b"] == ""
    assert df.iloc[4]["b"] == ""

    new_df = parse_cols(df)
    assert new_df.iloc[0]["a"] is None
    assert new_df.iloc[1]["a"] == 3
    assert new_df.iloc[2]["a"] == 4
    assert new_df.iloc[3]["a"] == 600
    assert new_df.iloc[4]["a"] == "62884541901610273646114"
    assert new_df.iloc[0]["b"] == "hi"
    assert new_df.iloc[1]["b"] == "  "
    assert new_df.iloc[2]["b"] is None
    assert new_df.iloc[3]["b"] is None
    assert new_df.iloc[4]["b"] is None
    assert len(new_df) == len(df)


def test__drop_unnamed() -> None:
    file_name = "unnamed.csv"
    df = _get_csv_data(file_name)
    cols = ["Unnamed: 0", "locator", "normalization_factor", "points"]
    assert df.columns.values.tolist() == cols

    new_df = drop_unnamed(df)
    new_cols = ["locator", "normalization_factor", "points"]
    assert new_df.columns.values.tolist() == new_cols
    assert len(new_df) == len(df)


def test__drop_invalid_ground_truth() -> None:
    file_name = "invalid_json_format.csv"
    df = _get_csv_data(file_name)
    cols = ["ground_truth", "metadata", "data_type", "locator"]
    assert df.columns.values.tolist() == cols

    new_df = drop_invalid_ground_truth(df)
    new_cols = ["metadata", "data_type", "locator"]
    assert new_df.columns.values.tolist() == new_cols
    assert len(new_df) == len(df)


def test__drop_invalid_metadata() -> None:
    file_name = "invalid_json_format.csv"
    df = _get_csv_data(file_name)
    cols = ["ground_truth", "metadata", "data_type", "locator"]
    assert df.columns.values.tolist() == cols

    new_df = drop_invalid_metadata(df)
    new_cols = ["ground_truth", "data_type", "locator"]
    assert new_df.columns.values.tolist() == new_cols
    assert len(new_df) == len(df)
