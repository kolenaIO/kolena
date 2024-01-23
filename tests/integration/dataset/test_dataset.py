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

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kolena._api.v2.dataset import CommitData
from kolena.dataset import download_dataset
from kolena.dataset import upload_dataset
from kolena.dataset.dataset import _fetch_dataset_history
from kolena.errors import NotFoundError
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import LabeledBoundingBox
from tests.integration.helper import fake_locator
from tests.integration.helper import with_test_prefix


TEST_DATASET_HISTORY_NAME = with_test_prefix(f"{__file__}::test__dataset_history")
TEST_DATASET_HISTORY_VERSIONS = 10


def _assert_frame_equal(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    assert_frame_equal(df1, df2, check_dtype=False)


def test__upload_dataset__empty() -> None:
    name = with_test_prefix(f"{__file__}::test__uplaod_dataset__empty")
    upload_dataset(name, pd.DataFrame(columns=["locator"]), id_fields=["locator"])

    assert download_dataset(name).empty


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
    _assert_frame_equal(loaded_datapoints, expected)

    # update dataset
    datapoints_updated = pd.DataFrame(datapoints[:5] + datapoints[7:15], columns=columns)
    upload_dataset(name, datapoints_updated, id_fields=["locator"])

    loaded_datapoints = download_dataset(name).sort_values("width", ignore_index=True).reindex(columns=columns)
    expected = pd.DataFrame(expected_datapoints[:5] + expected_datapoints[7:15], columns=columns)
    _assert_frame_equal(loaded_datapoints, expected)


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
    _assert_frame_equal(loaded_datapoints, expected)

    # update dataset
    datapoints_updated = pd.DataFrame(datapoints[:5] + datapoints[7:15], columns=columns)
    upload_dataset(name, datapoints_updated, id_fields=["locator"])

    loaded_datapoints = download_dataset(name).sort_values("width", ignore_index=True).reindex(columns=columns)
    expected = pd.DataFrame(expected_datapoints[:5] + expected_datapoints[7:15], columns=columns)
    _assert_frame_equal(loaded_datapoints, expected)


def test__upload_dataset__composite() -> None:
    name = with_test_prefix(f"{__file__}::test__upload_dataset__composite")
    datapoints = [
        {
            "a.text": "Something " * i,
            "b.text": "Something else " * i,
            "a.word_count": 1 * i,
            "b.word_count": 2 * i,
            "a.char_length": 10 * i,
            "b.char_length": 15 * i,
            "c": dict(text="nested " * i, word_count=3 * i, char_length=7 * i),
            "total_word_count": 3 * i,
            "total_char_length": 25 * i,
            "word_count_diff": i,
            "char_length_diff": 5 * i,
        }
        for i in range(1, 20)
    ]
    columns = datapoints[0].keys()
    id_fields = ["a.text"]

    df = pd.DataFrame(datapoints[:10], columns=columns)
    upload_dataset(name, df, id_fields=id_fields)

    loaded_datapoints = download_dataset(name)
    loaded_datapoints = loaded_datapoints.sort_values("total_word_count", ignore_index=True).reindex(columns=columns)
    _assert_frame_equal(df, loaded_datapoints)

    # update dataset
    df = pd.DataFrame(datapoints[:5] + datapoints[7:15], columns=columns)
    upload_dataset(name, df, id_fields=id_fields)

    loaded_datapoints = (
        download_dataset(name).sort_values("total_word_count", ignore_index=True).reindex(columns=columns)
    )
    _assert_frame_equal(df, loaded_datapoints)


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
        _assert_frame_equal(loaded_datapoints, expected_datapoints)


def test__download_dataset__commit_not_exist(with_dataset_commits: Tuple[int, List[CommitData]]) -> None:
    # check download_dataset with a commit that does not exist
    with pytest.raises(NotFoundError):
        download_dataset(TEST_DATASET_HISTORY_NAME, commit="non-existent-commit")
