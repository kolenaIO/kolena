# Copyright 2021-2025 Kolena Inc.
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
import uuid

import numpy as np
import pandas as pd
import pytest

from kolena._experimental.search import upload_embeddings
from kolena.dataset import upload_dataset
from kolena.dataset.embeddings import _upload_dataset_embeddings
from kolena.dataset.embeddings import download_dataset_embeddings
from kolena.dataset.embeddings import get_dataset_embedding_keys
from kolena.errors import InputValidationError
from kolena.errors import NotFoundError
from kolena.workflow import define_workflow
from kolena.workflow import GroundTruth
from kolena.workflow import Image
from kolena.workflow import Inference
from tests.integration.helper import fake_random_locator
from tests.integration.helper import with_test_prefix

DUMMY_WORKFLOW_NAME = "Dummy Workflow 🤖"

DUMMY_WORKFLOW, TestCase, TestSuite, Model = define_workflow(
    name=DUMMY_WORKFLOW_NAME,
    test_sample_type=Image,
    ground_truth_type=GroundTruth,
    inference_type=Inference,
)

N_DATAPOINTS = 20


def is_embedding_df_equal(
    df_uploaded: pd.DataFrame,
    df_downloaded: pd.DataFrame,
    sort_column: str,
    columns: list[str],
) -> bool:
    if len(df_uploaded) != len(df_downloaded):
        return False
    df_uploaded_sorted = df_uploaded.sort_values(by=sort_column).reset_index(drop=True)[columns]
    df_downloaded_sorted = df_downloaded.sort_values(by=sort_column).reset_index(drop=True)[columns]
    for row_ind in range(len(df_uploaded)):
        embedding_uploaded = df_uploaded_sorted["embedding"].iloc[row_ind]
        embedding_downloaded = df_downloaded_sorted["embedding"].iloc[row_ind]
        if len(embedding_uploaded) != len(embedding_downloaded) or not np.allclose(
            embedding_uploaded,
            embedding_downloaded,
        ):
            return False
    return True


@pytest.fixture(scope="module", autouse=True)
def test_sample_locator() -> str:
    test_case_name = with_test_prefix(f"{__file__} test_upload_embeddings")
    locator = fake_random_locator()
    TestCase.create(test_case_name, test_samples=[(Image(locator=locator), GroundTruth())])
    return locator


@pytest.mark.parametrize(
    "embedding",
    [
        np.array([1, 2, 3, 4], dtype=np.int32),
        np.array([1, 2, 3, 4], dtype=np.float64),
        np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64),
        np.array([], dtype=np.float16),
    ],
)
def test__upload_embeddings(embedding: np.ndarray, test_sample_locator: str) -> None:
    upload_embeddings(
        key="s3://model-bucket/embeddings-model.pt",
        embeddings=[(test_sample_locator, embedding)],
    )


@pytest.mark.parametrize(
    "embedding",
    [
        np.array([], dtype=str),
        np.array([b"1", b"2"]),
        np.array([1, "2"]),
    ],
)
def test__upload_embeddings__bad_embedding(embedding: np.ndarray) -> None:
    locator = fake_random_locator()
    with pytest.raises(InputValidationError):
        upload_embeddings(
            key="s3://model-bucket/embeddings-model.pt",
            embeddings=[(locator, embedding)],
        )


@pytest.fixture(scope="module", autouse=True)
def dataset_name() -> str:
    name = with_test_prefix(f"{__file__}::test__embedding_dataset {uuid.uuid4()}")  # noqa: E231
    datapoints = [dict(locator=f"locator-{i}", value=i) for i in range(N_DATAPOINTS)]
    upload_dataset(name, pd.DataFrame(datapoints), id_fields=["locator"])
    return name


@pytest.mark.parametrize(
    "embedding",
    [
        np.array([1, 2, 3, 4], dtype=np.int32),
        np.array([1, 2, 3, 4], dtype=np.float64),
        np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64),
        np.array([], dtype=np.float16),
    ],
)
def test__upload_dataset_embeddings(embedding: np.ndarray, dataset_name: str) -> None:
    locator_column = "locator"
    columns = ["locator", "embedding"]

    key_1 = "s3://model-bucket/embeddings-model.pt"
    df_embedding_1 = pd.DataFrame(
        {locator_column: [f"locator-{i}" for i in range(N_DATAPOINTS)], "embedding": [embedding] * N_DATAPOINTS},
    )
    _upload_dataset_embeddings(
        dataset_name,
        key=key_1,
        df_embedding=df_embedding_1,
        run_embedding_reduction_pipeline=False,
    )
    assert key_1 in get_dataset_embedding_keys(dataset_name)
    df_embedding_downloaded_1 = download_dataset_embeddings(dataset_name, key_1)
    assert is_embedding_df_equal(
        df_uploaded=df_embedding_1,
        df_downloaded=df_embedding_downloaded_1,
        sort_column=locator_column,
        columns=columns,
    )

    key_2 = "my_model-left_image"
    df_embedding_2 = pd.DataFrame(
        {locator_column: [f"locator-{i}" for i in range(N_DATAPOINTS)], "embedding": [embedding * 2] * N_DATAPOINTS},
    )
    _upload_dataset_embeddings(
        dataset_name,
        key=key_2,
        df_embedding=df_embedding_2,
        run_embedding_reduction_pipeline=False,
    )
    assert key_2 in get_dataset_embedding_keys(dataset_name)
    df_embedding_downloaded_2 = download_dataset_embeddings(dataset_name, key_2)
    assert is_embedding_df_equal(
        df_uploaded=df_embedding_2,
        df_downloaded=df_embedding_downloaded_2,
        sort_column=locator_column,
        columns=columns,
    )

    key_3 = "my_model-right_image"
    df_embedding_3 = pd.DataFrame(
        {locator_column: [f"locator-{i}" for i in range(N_DATAPOINTS)], "embedding": [embedding * 3] * N_DATAPOINTS},
    )
    _upload_dataset_embeddings(
        dataset_name,
        key=key_3,
        df_embedding=df_embedding_3,
        run_embedding_reduction_pipeline=False,
    )
    assert key_3 in get_dataset_embedding_keys(dataset_name)
    df_embedding_downloaded_3 = download_dataset_embeddings(dataset_name, key_3)
    assert is_embedding_df_equal(
        df_uploaded=df_embedding_3,
        df_downloaded=df_embedding_downloaded_3,
        sort_column=locator_column,
        columns=columns,
    )


def test__upload_dataset_embeddings__partial_dataset(dataset_name: str) -> None:
    key = "s3://model-bucket/partial-embeddings-model.pt"
    locator_column = "locator"
    columns = ["locator", "embedding"]
    df_embedding = pd.DataFrame(
        {
            "locator": [f"locator-{i}" for i in range(N_DATAPOINTS // 2)],
            "embedding": [np.array([1, 2, 3, 4], dtype=np.int32)] * (N_DATAPOINTS // 2),
        },
    )
    _upload_dataset_embeddings(
        dataset_name,
        key=key,
        df_embedding=df_embedding,
        run_embedding_reduction_pipeline=False,
    )

    assert key in get_dataset_embedding_keys(dataset_name)
    df_embedding_downloaded_3 = download_dataset_embeddings(dataset_name, key)
    assert is_embedding_df_equal(
        df_uploaded=df_embedding,
        df_downloaded=df_embedding_downloaded_3,
        sort_column=locator_column,
        columns=columns,
    )


def test__upload_dataset_embeddings__dataset_does_not_exist() -> None:
    with pytest.raises(NotFoundError):
        _upload_dataset_embeddings(
            dataset_name=f"test__embedding_dataset_does_not_exist {uuid.uuid4()}",  # noqa: E231
            key="s3://model-bucket/embeddings-model.pt",
            df_embedding=pd.DataFrame(
                {"locator": [], "embedding": []},
            ),
            run_embedding_reduction_pipeline=False,
        )


def test__get_dataset_embedding_keys__dataset_does_not_exist() -> None:
    with pytest.raises(NotFoundError):
        get_dataset_embedding_keys(
            dataset_name=f"test__get_dataset_embedding_keys__dataset_does_not_exist {uuid.uuid4()}",  # noqa: E231
        )


def test__download_dataset_embeddings__dataset_does_not_exist() -> None:
    with pytest.raises(NotFoundError):
        download_dataset_embeddings(
            dataset_name=f"test__download_dataset_embeddings__dataset_does_not_exist {uuid.uuid4()}",  # noqa: E231
            key="some-key",
        )


def test__download_dataset_embeddings__key_does_not_exist(dataset_name: str) -> None:
    with pytest.raises(NotFoundError):
        download_dataset_embeddings(
            dataset_name=dataset_name,
            key=f"test__download_dataset_embeddings__key_does_not_exist {uuid.uuid4()}",  # noqa: E231
        )


def test__upload_dataset_embeddings__id_fields_mismatch(dataset_name: str) -> None:
    with pytest.raises(InputValidationError):
        _upload_dataset_embeddings(
            dataset_name,
            key="s3://model-bucket/embeddings-model.pt",
            df_embedding=pd.DataFrame(
                {"value": [], "embedding": []},
            ),
            run_embedding_reduction_pipeline=False,
        )


@pytest.mark.parametrize(
    "embedding",
    [
        np.array([], dtype=str),
        np.array([b"1", b"2"]),
        np.array([1, "2"]),
    ],
)
def test__upload_dataset_embeddings__bad_embedding(embedding: np.ndarray, dataset_name: str) -> None:
    with pytest.raises(InputValidationError):
        _upload_dataset_embeddings(
            dataset_name,
            key="s3://model-bucket/embeddings-model.pt",
            df_embedding=pd.DataFrame(
                {"locator": [f"locator-{i}" for i in range(N_DATAPOINTS)], "embedding": [embedding] * N_DATAPOINTS},
            ),
            run_embedding_reduction_pipeline=False,
        )


def test__upload_dataset_embeddings__embedding_different_sizes(dataset_name: str) -> None:
    embedding = [np.array([1] * i, dtype=np.float64) for i in range(N_DATAPOINTS)]
    with pytest.raises(InputValidationError):
        _upload_dataset_embeddings(
            dataset_name,
            key="s3://model-bucket/embeddings-model.pt",
            df_embedding=pd.DataFrame(
                {"locator": [f"locator-{i}" for i in range(N_DATAPOINTS)], "embedding": embedding},
            ),
            run_embedding_reduction_pipeline=False,
        )
