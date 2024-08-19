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
from typing import Iterator
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from kolena._api.v2.dataset import CommitData
from kolena.dataset import download_dataset
from kolena.dataset import list_datasets
from kolena.dataset import upload_dataset
from kolena.dataset.dataset import _fetch_dataset_history
from kolena.dataset.dataset import _load_dataset_metadata
from kolena.errors import InputValidationError
from kolena.errors import NotFoundError
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import LabeledBoundingBox
from tests.integration.helper import assert_frame_equal
from tests.integration.helper import fake_locator
from tests.integration.helper import upload_extracted_properties
from tests.integration.helper import with_test_prefix


TEST_DATASET_HISTORY_NAME = with_test_prefix(f"{__file__}::test__dataset_history")
TEST_DATASET_HISTORY_VERSIONS = 10


def test__load_dataset_metadata_dataset__not_exist() -> None:
    name = with_test_prefix(f"{__file__}::test__load_dataset_metadata_dataset__not_exist")
    with pytest.raises(NotFoundError):
        _load_dataset_metadata(name)
    assert _load_dataset_metadata(name, raise_error_if_not_found=False) is None


def test__upload_dataset__empty() -> None:
    name = with_test_prefix(f"{__file__}::test__upload_dataset__empty")
    with pytest.raises(InputValidationError):
        upload_dataset(name, pd.DataFrame(columns=["locator"]), id_fields=["locator"])


def test__upload_dataset__empty_iterator() -> None:
    name = with_test_prefix(f"{__file__}::test__upload_dataset__empty_iterator")
    with pytest.raises(InputValidationError):
        upload_dataset(name, batch_iterator(pd.DataFrame(columns=["locator"])), id_fields=["locator"])


def test__upload_dataset__empty_name() -> None:
    name = " "
    with pytest.raises(InputValidationError):
        upload_dataset(name, pd.DataFrame([dict(locator="0.jpg")], columns=["locator"]), id_fields=["locator"])


def test__list_datasets() -> None:
    name = with_test_prefix(f"{__file__}::test__list_datasets")
    upload_dataset(name, pd.DataFrame([dict(locator="0.jpg")], columns=["locator"]), id_fields=["locator"])
    assert name in list_datasets()


def test__upload_dataset() -> None:
    name = with_test_prefix(f"{__file__}::test__upload_dataset")
    datapoints = [
        dict(
            locator=fake_locator(i, name),
            width=i + 500,
            height=i + 400,
            city=random.choice(["new york", "waterloo"]),
            bboxes=[
                LabeledBoundingBox(label="cat", top_left=[i, i], bottom_right=[i + 10, i + 10]),
                LabeledBoundingBox(label="dog", top_left=[i + 5, i + 5], bottom_right=[i + 20, i + 20]),
            ],
        )
        for i in range(20)
    ]
    expected_datapoints = [
        dict(
            locator=dp["locator"],
            width=dp["width"],
            height=dp["height"],
            city=dp["city"],
            bboxes=[
                BoundingBox(label=bbox.label, top_left=bbox.top_left, bottom_right=bbox.bottom_right)
                for bbox in dp["bboxes"]
            ],
        )
        for dp in datapoints
    ]
    columns = ["locator", "width", "height", "city", "bboxes"]

    upload_dataset(name, pd.DataFrame(datapoints[:10], columns=columns), id_fields=["locator"])

    loaded_datapoints = download_dataset(name).sort_values("width", ignore_index=True).reindex(columns=columns)
    expected = pd.DataFrame(expected_datapoints[:10], columns=columns)
    assert_frame_equal(loaded_datapoints, expected)

    # update dataset
    datapoints_updated = pd.DataFrame(datapoints[:5] + datapoints[7:15], columns=columns)
    upload_dataset(name, datapoints_updated, id_fields=["locator"])

    loaded_datapoints = download_dataset(name).sort_values("width", ignore_index=True).reindex(columns=columns)
    expected = pd.DataFrame(expected_datapoints[:5] + expected_datapoints[7:15], columns=columns)
    assert_frame_equal(loaded_datapoints, expected)


def batch_iterator(df: pd.DataFrame, batch_size=5) -> Iterator[pd.DataFrame]:
    for i in range(0, len(df), batch_size):
        yield df.iloc[i : i + batch_size]


def test__upload_dataset_chunks() -> None:
    name = with_test_prefix(f"{__file__}::test__upload_dataset")
    datapoints = [
        dict(
            locator=fake_locator(i, name),
            width=i + 500,
            height=i + 400,
            city=random.choice(["new york", "waterloo"]),
            bboxes=[
                LabeledBoundingBox(label="cat", top_left=[i, i], bottom_right=[i + 10, i + 10]),
                LabeledBoundingBox(label="dog", top_left=[i + 5, i + 5], bottom_right=[i + 20, i + 20]),
            ],
        )
        for i in range(20)
    ]
    expected_datapoints = [
        dict(
            locator=dp["locator"],
            width=dp["width"],
            height=dp["height"],
            city=dp["city"],
            bboxes=[
                BoundingBox(label=bbox.label, top_left=bbox.top_left, bottom_right=bbox.bottom_right)
                for bbox in dp["bboxes"]
            ],
        )
        for dp in datapoints
    ]
    columns = ["locator", "width", "height", "city", "bboxes"]

    upload_dataset(name, batch_iterator(pd.DataFrame(datapoints[:10], columns=columns)), id_fields=["locator"])

    loaded_datapoints = download_dataset(name).sort_values("width", ignore_index=True).reindex(columns=columns)
    expected = pd.DataFrame(expected_datapoints[:10], columns=columns)
    assert_frame_equal(loaded_datapoints, expected)

    # update dataset
    datapoints_updated = pd.DataFrame(datapoints[:5] + datapoints[7:15], columns=columns)
    upload_dataset(name, datapoints_updated, id_fields=["locator"])

    loaded_datapoints = download_dataset(name).sort_values("width", ignore_index=True).reindex(columns=columns)
    expected = pd.DataFrame(expected_datapoints[:5] + expected_datapoints[7:15], columns=columns)
    assert_frame_equal(loaded_datapoints, expected)


def test__download_dataset__not_exist() -> None:
    name = with_test_prefix(f"{__file__}::test__download_dataset__not_exist")
    with pytest.raises(NotFoundError):
        download_dataset(name)


@pytest.fixture(scope="module")
def with_dataset_commits() -> Tuple[int, List[CommitData]]:
    for version in range(TEST_DATASET_HISTORY_VERSIONS):
        # remove all previous datapoints and add (version + 1) new datapoints in each iteration
        datapoints = [dict(locator=f"{version}-{i}") for i in range(version + 1)]
        upload_dataset(TEST_DATASET_HISTORY_NAME, pd.DataFrame(datapoints), id_fields=["locator"])
    return _fetch_dataset_history(TEST_DATASET_HISTORY_NAME)


def test__download_commits(with_dataset_commits: Tuple[int, List[CommitData]]) -> None:
    # check download_commits without optional args
    total_commits, commits = with_dataset_commits
    assert total_commits == TEST_DATASET_HISTORY_VERSIONS
    previous_commit_time = -1
    for version, commit in enumerate(commits):
        assert commit.timestamp >= previous_commit_time
        assert commit.n_removed == version
        assert commit.n_added == version + 1
        previous_commit_time = commit.timestamp


def test__download_commits__pagination(with_dataset_commits: Tuple[int, List[CommitData]]) -> None:
    # check download_commits with desc arg
    total_commits, commits = with_dataset_commits
    total_commits_pagination, commits_pagination = _fetch_dataset_history(TEST_DATASET_HISTORY_NAME, page_size=1)
    assert total_commits_pagination == total_commits
    assert commits_pagination == commits


def test__download_commits__desc(with_dataset_commits: Tuple[int, List[CommitData]]) -> None:
    # check download_commits with desc arg
    total_commits, commits = with_dataset_commits
    total_commits_desc, commits_desc = _fetch_dataset_history(TEST_DATASET_HISTORY_NAME, descending=True)
    assert total_commits_desc == total_commits
    assert commits_desc == commits[::-1]


def test__download_commits__limit(with_dataset_commits: Tuple[int, List[CommitData]]) -> None:
    # check download_commits with limit arg
    total_commits, commits = with_dataset_commits
    total_commits_limit, commits_limit = _fetch_dataset_history(TEST_DATASET_HISTORY_NAME, limit=10)
    assert total_commits_limit == total_commits
    assert commits_limit == commits[:10]


def test__download_commits__limit_more_than_versions(with_dataset_commits: Tuple[int, List[CommitData]]) -> None:
    # check download_commits with limit arg greater than number of revisions
    total_commits, commits = with_dataset_commits
    total_commits_limit, commits_limit = _fetch_dataset_history(
        TEST_DATASET_HISTORY_NAME,
        limit=TEST_DATASET_HISTORY_VERSIONS + 1,
    )
    assert total_commits_limit == total_commits
    assert len(commits) == TEST_DATASET_HISTORY_VERSIONS
    assert commits_limit == commits


def test__download_commits__desc_limit(with_dataset_commits: Tuple[int, List[CommitData]]) -> None:
    # check download_commits with both desc and limit args
    total_commits, commits = with_dataset_commits
    total_commits_desc_limit, commits_desc_limit = _fetch_dataset_history(
        TEST_DATASET_HISTORY_NAME,
        descending=True,
        limit=10,
    )
    assert total_commits_desc_limit == total_commits
    assert commits_desc_limit == commits[-1:-11:-1]


def test__download_dataset__versions(with_dataset_commits: Tuple[int, List[CommitData]]) -> None:
    # check download_dataset with commit arg
    total_commits, commits = with_dataset_commits
    for version, commit in enumerate(commits):
        loaded_datapoints = download_dataset(TEST_DATASET_HISTORY_NAME, commit=commit.commit).sort_values(
            "locator",
            ignore_index=True,
        )
        expected_datapoints = pd.DataFrame([dict(locator=f"{version}-{i}") for i in range(version + 1)]).sort_values(
            "locator",
            ignore_index=True,
        )
        assert_frame_equal(loaded_datapoints, expected_datapoints)


def test__download_dataset__with_property() -> None:
    name = with_test_prefix(f"{__file__}::test__download_dataset__with_property")
    datapoints = pd.DataFrame(
        [
            dict(
                locator=fake_locator(i, name),
                text=f"dummy text {i}",
                id=i,
            )
            for i in range(20)
        ],
    )
    extracted_property = [{"llm": {"summary": f"dummy text {i}"}} for i in range(20)]
    upload_dataset(name, datapoints, id_fields=["locator"])
    dataset_id = _load_dataset_metadata(name).id
    datapoints["extracted"] = extracted_property
    upload_extracted_properties(
        dataset_id,
        datapoints,
        id_fields=["locator"],
    )

    loaded_datapoints = download_dataset(name, include_extracted_properties=True).sort_values("id", ignore_index=True)
    datapoints["kolena_llm_prompt_extraction"] = [prop["llm"] for prop in extracted_property]
    datapoints.drop(columns=["extracted"], inplace=True)
    pd.testing.assert_frame_equal(
        loaded_datapoints,
        datapoints[loaded_datapoints.columns],
        check_like=True,
        check_dtype=False,
    )


def test__download_dataset__commit_not_exist(with_dataset_commits: Tuple[int, List[CommitData]]) -> None:
    # check download_dataset with a commit that does not exist
    with pytest.raises(NotFoundError):
        download_dataset(TEST_DATASET_HISTORY_NAME, commit="non-existent-commit")


def test__download_dataset__preserve_none() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__download_dataset__preserve_none")

    data = [{"a": None, "id": 1}, {"a": float("inf"), "id": 2}, {"a": np.nan, "id": 3}, {"a": 42, "id": 4}]
    df_dp = pd.DataFrame.from_dict(data, dtype=object)
    assert df_dp["a"][0] is None
    assert np.isinf(df_dp["a"][1])
    assert np.isnan(df_dp["a"][2])
    upload_dataset(dataset_name, df_dp, id_fields=["id"])

    fetched_df_dp = download_dataset(dataset_name)

    assert_frame_equal(fetched_df_dp, df_dp)
    assert fetched_df_dp["a"][0] is None
    assert np.isinf(fetched_df_dp["a"][1])
    assert np.isnan(fetched_df_dp["a"][2])
