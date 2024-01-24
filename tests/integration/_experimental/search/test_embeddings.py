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
import uuid

import numpy as np
import pandas as pd
import pytest

from kolena._experimental.search import upload_dataset_embeddings
from kolena._experimental.search import upload_embeddings
from kolena.dataset import upload_dataset
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


@pytest.mark.parametrize(
    "embedding",
    [
        np.array([1, 2, 3, 4], dtype=np.int32),
        np.array([1, 2, 3, 4], dtype=np.float64),
        np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64),
        np.array([], dtype=np.float16),
    ],
)
def test__upload_embeddings(embedding: np.ndarray) -> None:
    test_case_name = with_test_prefix(f"{__file__} test_upload_embeddings {uuid.uuid4()}")
    locator = fake_random_locator()
    TestCase.create(test_case_name, test_samples=[(Image(locator=locator), GroundTruth())])
    upload_embeddings(
        key="s3://model-bucket/embeddings-model.pt",
        embeddings=[(locator, embedding)],
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
    name = with_test_prefix(f"{__file__}::test__embedding_dataset {uuid.uuid4()}")
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
    upload_dataset_embeddings(
        dataset_name,
        key="s3://model-bucket/embeddings-model.pt",
        df_embedding=pd.DataFrame(
            {"locator": [f"locator-{i}" for i in range(N_DATAPOINTS)], "embedding": [embedding] * N_DATAPOINTS},
        ),
    )


def test__upload_dataset_embeddings__dataset_does_not_exist() -> None:
    with pytest.raises(NotFoundError):
        upload_dataset_embeddings(
            dataset_name=f"{__file__}::test__embedding_dataset_does_not_exist {uuid.uuid4()}",
            key="s3://model-bucket/embeddings-model.pt",
            df_embedding=pd.DataFrame(
                {"locator": [], "embedding": []},
            ),
        )


def test__upload_dataset_embeddings__id_fields_mismatch(dataset_name: str) -> None:
    with pytest.raises(InputValidationError):
        upload_dataset_embeddings(
            dataset_name,
            key="s3://model-bucket/embeddings-model.pt",
            df_embedding=pd.DataFrame(
                {"value": [], "embedding": []},
            ),
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
        upload_dataset_embeddings(
            dataset_name,
            key="s3://model-bucket/embeddings-model.pt",
            df_embedding=pd.DataFrame(
                {"locator": [f"locator-{i}" for i in range(N_DATAPOINTS)], "embedding": [embedding] * N_DATAPOINTS},
            ),
        )
