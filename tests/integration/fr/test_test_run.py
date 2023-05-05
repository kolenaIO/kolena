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
from typing import List
from typing import Optional
from typing import Tuple
from unittest.mock import patch

import numpy as np
import pandas as pd
import pandas.testing
import pytest

from kolena._api.v1.fr import TestRun as TestRunAPI
from kolena.errors import InputValidationError
from kolena.errors import RemoteError
from kolena.fr import EmbeddingDataFrame
from kolena.fr import ImageResultDataFrame
from kolena.fr import InferenceModel
from kolena.fr import Model
from kolena.fr import PairDataFrame
from kolena.fr import PairResultDataFrame
from kolena.fr import test
from kolena.fr import TestCase
from kolena.fr import TestImages
from kolena.fr import TestRun
from kolena.fr import TestSuite
from kolena.fr.datatypes import TestCaseRecord
from tests.integration.fr.conftest import TestData
from tests.integration.fr.helper import generate_image_results
from tests.integration.helper import with_test_prefix


def assert_similarity_equal(df: pd.DataFrame, val: float) -> None:
    assert all([record.similarity == val for record in df.itertuples()])


@pytest.fixture(scope="session")
def fr_test_run_core(fr_test_suites: List[TestSuite]) -> Tuple[Model, TestSuite]:
    model = Model.create(with_test_prefix(f"{__file__}::test__test_run_core model"), metadata={})
    return model, fr_test_suites[0]


@pytest.fixture
def fr_test_run(fr_test_run_core: Tuple[Model, TestSuite]) -> TestRun:
    return TestRun(*fr_test_run_core)


@pytest.fixture(scope="session")
def fr_multi_face_test_run(fr_test_suites: List[TestSuite]) -> Tuple[Model, TestSuite]:
    model = Model.create(with_test_prefix(f"{__file__}::test__multi_face model"), metadata={})
    test_suite = fr_test_suites[0]
    return model, test_suite


def generate_pair_results(df_emb: EmbeddingDataFrame, df_pair: PairDataFrame) -> PairResultDataFrame:
    similarities: List[Optional[float]] = []
    for pair_record in df_pair.itertuples():
        df_emb_a = df_emb[df_emb["image_id"] == pair_record.image_a_id]
        assert len(df_emb_a) == 1
        df_emb_b = df_emb[df_emb["image_id"] == pair_record.image_b_id]
        assert len(df_emb_b) == 1
        emb_a, emb_b = df_emb_a.iloc[0]["embedding"], df_emb_b.iloc[0]["embedding"]
        # properly populate pair failure when one or more of the images lacks an embedding
        similarities.append(np.random.rand() if emb_a is not None and emb_b is not None else None)
    return pd.DataFrame(dict(image_pair_id=df_pair["image_pair_id"].tolist(), similarity=similarities))


#
# Interacting with a TestRun is naturally a sequenced operation -- here each test depends on the next test and likely
# uses some of the same functionality
#


# Note: create_or_retrieve is deprecated
def test__create_or_retrieve(fr_test_run_core: Tuple[Model, TestSuite]) -> None:
    model, test_suite = fr_test_run_core
    input_test_suite_id = test_suite._id

    test_run_created = TestRun.create_or_retrieve(model, test_suite)
    assert test_run_created.data.model.id == model.data.id
    created_test_suite_ids = [test_suite.id for test_suite in test_run_created.data.test_suites]
    assert len(created_test_suite_ids) == 1
    assert created_test_suite_ids[0] == input_test_suite_id

    test_run_retrieved = TestRun.create_or_retrieve(model, test_suite)
    assert test_run_retrieved.data.id == test_run_created.data.id
    assert test_run_retrieved.data.model.id == model.data.id
    retrieved_test_suite_ids = [test_suite.id for test_suite in test_run_retrieved.data.test_suites]
    assert len(retrieved_test_suite_ids) == 1
    assert retrieved_test_suite_ids[0] == input_test_suite_id


@pytest.mark.depends(on=["test__create_or_retrieve"])
def test__load_remaining_images(fr_test_data: TestData, fr_test_run: TestRun) -> None:
    image_pairs = fr_test_data.image_pairs
    # load all images in test suite, containing only unique entries (no duplicates from images in multiple test cases)
    expected_image_pairs = [
        image_pairs[0],
        image_pairs[1],
        image_pairs[2],
        image_pairs[8],
        image_pairs[9],
    ]
    expected_image_locators = sorted(list({loc for pair in expected_image_pairs for loc in pair[:2]}))
    df_remaining_images_expected = pd.DataFrame(dict(locator=expected_image_locators))

    df_remaining_images_actual = (
        fr_test_run.load_remaining_images()
        .drop(columns="image_id")
        .sort_values(
            by="locator",
            ignore_index=True,
        )
    )
    pandas.testing.assert_frame_equal(df_remaining_images_actual, df_remaining_images_expected)

    # fetching again should retrieve the same data if no results were uploaded
    df_remaining_images_actual = (
        fr_test_run.load_remaining_images(batch_size=500)
        .drop(columns="image_id")
        .sort_values(
            by="locator",
            ignore_index=True,
        )
    )
    pandas.testing.assert_frame_equal(df_remaining_images_actual, df_remaining_images_expected)

    batch_size = 2
    df_remaining_images_actual = fr_test_run.load_remaining_images(batch_size=2)
    assert len(df_remaining_images_actual) == batch_size

    # zero-size batches are not allowed
    with pytest.raises(InputValidationError):
        fr_test_run.load_remaining_images(batch_size=0)


@pytest.mark.depends(on=["test__load_remaining_images"])
def test__load_remaining_pairs__premature(fr_test_run: TestRun) -> None:
    # loading pairs should be blocked until all images are processed
    with pytest.raises(RemoteError):
        fr_test_run.load_remaining_pairs()


@pytest.mark.depends(on=["test__load_remaining_images"])
def test__upload_image_results__validation(fr_test_run: TestRun) -> None:
    with pytest.raises(InputValidationError):
        # basic assertion that input DataFrame is validated
        fr_test_run.upload_image_results(pd.DataFrame(dict(garbage=[1])))

    with pytest.raises(InputValidationError):
        # zero-length frames are disallowed
        fr_test_run.upload_image_results(generate_image_results([]))

    with pytest.raises(InputValidationError):
        # assert that non-null is properly enforced
        df = generate_image_results(list(range(10)))
        df.at[0, "image_id"] = None
        fr_test_run.upload_image_results(df)

    with pytest.raises(RemoteError):
        # bogus IDs should fail during ingest
        id_offset = 99999
        fr_test_run.upload_image_results(generate_image_results(list(range(id_offset, id_offset + 10))))


@pytest.mark.depends(on=["test__upload_image_results__validation"])
def test__upload_image_results(fr_test_run: TestRun) -> None:
    df_remaining_images = fr_test_run.load_remaining_images(batch_size=2)
    df_image_results = generate_image_results(df_remaining_images["image_id"].tolist())

    n_uploaded = fr_test_run.upload_image_results(df_image_results)
    assert n_uploaded == len(df_remaining_images)

    with pytest.raises(RemoteError):
        # shouldn't be able to upload duplicate entries
        fr_test_run.upload_image_results(df_image_results)

    # test failure to enroll
    df_remaining_images = fr_test_run.load_remaining_images(batch_size=1)
    df_image_results = pd.DataFrame(
        dict(
            image_id=df_remaining_images["image_id"].tolist(),
            bounding_box=[None],
            landmarks_input_image=[None],
            landmarks=[None],
            quality_input_image=[None],
            quality=[None],
            acceptability=[None],
            fr_input_image=[None],
            embedding=[None],
            failure_reason=["failed for this reason"],
        ),
    )
    n_uploaded = fr_test_run.upload_image_results(df_image_results)
    assert n_uploaded == len(df_remaining_images)

    df_remaining_images = fr_test_run.load_remaining_images()
    df_image_results = generate_image_results(df_remaining_images["image_id"].tolist())
    n_uploaded = fr_test_run.upload_image_results(df_image_results)
    assert n_uploaded == len(df_image_results)

    # results have been received for the entire suite, no more images to fetch
    df_remaining_images = fr_test_run.load_remaining_images()
    assert len(df_remaining_images) == 0


@pytest.mark.depends(on=["test__upload_image_results__validation"])
def test__load_remaining_pairs(fr_test_data: TestData, fr_test_run: TestRun) -> None:
    data_sources = fr_test_data.data_sources
    images = TestImages.load(data_source=data_sources[0])
    image_pairs = fr_test_data.image_pairs
    test_suite_image_pairs = [
        image_pairs[0],
        image_pairs[1],
        image_pairs[2],
        image_pairs[8],
        image_pairs[9],
    ]
    images_dict = dict(zip(images.locator, images.image_id))
    image_pair_ids = sorted(
        [(images_dict[locator_a], images_dict[locator_b]) for locator_a, locator_b, is_same in test_suite_image_pairs],
    )

    columns = ["image_a_id", "image_b_id"]
    df_image_pair_expected = pd.DataFrame.from_records(
        image_pair_ids,
        columns=columns,
    ).sort_values(by=columns, ignore_index=True)
    df_embedding_actual, df_image_pair_actual = fr_test_run.load_remaining_pairs()
    pandas.testing.assert_frame_equal(
        df_image_pair_actual[columns].sort_values(by=columns, ignore_index=True),
        df_image_pair_expected,
    )

    embedding_image_ids = sorted(df_embedding_actual["image_id"].tolist())
    assert embedding_image_ids == sorted(images_dict.values())

    df_embedding_actual, df_image_pair_actual = fr_test_run.load_remaining_pairs(batch_size=1)
    assert len(df_image_pair_actual) == 1
    actual_image_ids = set(df_image_pair_actual["image_a_id"].tolist()).union(
        set(df_image_pair_actual["image_b_id"].tolist()),
    )
    assert len(df_embedding_actual[df_embedding_actual["image_id"].isin(actual_image_ids)]) == len(df_embedding_actual)


@pytest.mark.depends(on=["test__load_remaining_pairs"])
def test__upload_pair_results__validation(fr_test_run: TestRun) -> None:
    with pytest.raises(InputValidationError):
        # basic assertion that input DataFrame is validated
        fr_test_run.upload_pair_results(pd.DataFrame(dict(garbage=[1])))

    with pytest.raises(InputValidationError):
        # zero-length frames are disallowed
        fr_test_run.upload_pair_results(pd.DataFrame([], columns=["image_pair_id", "similarity"]))

    with pytest.raises(RemoteError):
        # bogus IDs should fail during ingest
        n_records = 10
        id_offset = 99999
        df = pd.DataFrame(
            dict(
                image_pair_id=list(range(id_offset, id_offset + n_records)),
                similarity=np.random.rand(n_records).astype(np.float32).tolist(),
            ),
        )
        fr_test_run.upload_pair_results(df)


@pytest.mark.depends(on=["test__upload_pair_results__validation"])
def test__upload_pair_results(fr_test_run: TestRun) -> None:
    df_embedding, df_image_pair = fr_test_run.load_remaining_pairs(batch_size=2)
    df_image_pair_results = generate_pair_results(df_embedding, df_image_pair)
    n_uploaded = fr_test_run.upload_pair_results(df_image_pair_results)
    assert n_uploaded == len(df_image_pair)

    with pytest.raises(RemoteError):
        # shouldn't be able to upload duplicate entries
        fr_test_run.upload_pair_results(df_image_pair_results)

    df_embedding, df_image_pair = fr_test_run.load_remaining_pairs(batch_size=1)
    df_image_pair_results = generate_pair_results(df_embedding, df_image_pair)
    n_uploaded = fr_test_run.upload_pair_results(df_image_pair_results)
    assert n_uploaded == len(df_image_pair)

    df_embedding, df_image_pair = fr_test_run.load_remaining_pairs(batch_size=2)
    fr_test_run.upload_pair_results(generate_pair_results(df_embedding, df_image_pair))

    # results have been received for the entire suite, no more pairs to fetch
    df_embedding, df_image_pair = fr_test_run.load_remaining_pairs()
    assert len(df_embedding) == 0
    assert len(df_image_pair) == 0


@pytest.mark.depends(on=["test__upload_pair_results"])
def test__noop(fr_test_run_core: Tuple[Model, TestSuite], fr_test_suites: List[TestSuite]) -> None:
    model = fr_test_run_core[0]
    test_suite = fr_test_suites[3]

    test_run = TestRun(model, test_suite)

    df_remaining_images = test_run.load_remaining_images()
    assert len(df_remaining_images) == 0
    df_embedding, df_image_pair = test_run.load_remaining_pairs()
    assert len(df_embedding) == 0
    assert len(df_image_pair) == 0


def test__test(fr_test_suites: List[TestSuite]) -> None:
    model = InferenceModel.create(
        with_test_prefix(f"{__file__}::test__test model"),
        extract=lambda _: np.random.rand(256).astype(np.float32),
        compare=lambda _, __: np.random.rand(1).astype(np.float32)[0],
        metadata={},
    )
    test_suites = [fr_test_suites[0], fr_test_suites[2]]
    test_run = test(model, test_suites[0])

    assert len(test_run.load_remaining_images()) == 0
    assert len(test_run.load_remaining_pairs()[1]) == 0


@pytest.mark.depends(on=["test__upload_pair_results"])
def test__test__reset(fr_models: List[Model], fr_test_suites: List[TestSuite]) -> None:
    model = fr_models[0]
    test_suites = [fr_test_suites[0], fr_test_suites[2]]
    other_name = with_test_prefix(f"{__file__}::test__test__reset other model")
    other_model = InferenceModel.create(
        other_name,
        extract=lambda _: None,
        compare=lambda _, __: 0.99,
        metadata={},
    )
    test(other_model, test_suites[0])

    reset_model = InferenceModel.load_by_name(
        model.data.name,
        extract=lambda _: None,
        compare=lambda _, __: 0.89,
    )
    test(reset_model, test_suites[0], reset=True)
    df_before_0 = reset_model.load_pair_results(test_suites[0])
    df_before_1 = reset_model.load_pair_results(test_suites[1])
    assert len(df_before_0) == 5
    # test suite A and B has one common image pair [2]
    # which is the reason why there are pair results for df_before_1 and df_after_1
    assert len(df_before_1) == 1
    assert_similarity_equal(df_before_0, 0.89)
    assert_similarity_equal(df_before_1, 0.89)
    assert_similarity_equal(other_model.load_pair_results(test_suites[0]), 0.99)


def test__test__mark_crashed(fr_test_suites: List[TestSuite]) -> None:
    def extract(locator: str) -> np.ndarray:
        raise RuntimeError(f"failed to process image: {locator}")

    model = InferenceModel.create(
        with_test_prefix(f"{__file__}::test__test__mark_crashed model"),
        extract=extract,
        compare=lambda _, __: np.random.rand(1).astype(np.float32)[0],
        metadata={},
    )
    test_suites = [fr_test_suites[0], fr_test_suites[2]]
    test_run = TestRun(model, test_suites[0])

    with patch("kolena.fr.test_run.report_crash") as patched:
        with pytest.raises(RuntimeError):
            test(model, test_suites[0])

    patched.assert_called_once_with(test_run._id, TestRunAPI.Path.MARK_CRASHED)


def test__multi_face(fr_multi_face_test_run: Tuple[Model, TestSuite]) -> None:
    model, test_suite = fr_multi_face_test_run

    test_run = TestRun(model, test_suite)

    df_image = test_run.load_remaining_images()
    n_images = len(df_image)
    df_image_result = pd.DataFrame(
        dict(
            image_id=df_image["image_id"].tolist(),
            bounding_box=[np.random.rand(4).astype(np.float32) for _ in range(n_images)],
            landmarks=[np.random.rand(10).astype(np.float32) for _ in range(n_images)],
            quality=np.random.rand(n_images).astype(np.float64).tolist(),
            acceptability=np.random.rand(n_images).astype(np.float64).tolist(),
            embedding=[np.random.rand(256).astype(np.float32) for _ in range(n_images)],
        ),
    )
    df_image_result_doubled = pd.concat([df_image_result, df_image_result], ignore_index=True)
    n_uploaded = test_run.upload_image_results(ImageResultDataFrame(df_image_result_doubled))
    assert n_uploaded == len(df_image_result_doubled)
    assert len(test_run.load_remaining_images()) == 0

    df_embedding, df_pair = test_run.load_remaining_pairs()

    assert all(np.shape(record.embedding) == (2, 256) for record in df_embedding.itertuples())

    n_pairs = len(df_pair)
    df_pair_result = pd.DataFrame(
        dict(
            image_pair_id=df_pair["image_pair_id"].tolist() * 4,
            similarity=[np.random.rand(1)[0] for _ in range(n_pairs * 4)],
            embedding_a_index=[0, 1, 0, 1] * n_pairs,
            embedding_b_index=[0, 0, 1, 1] * n_pairs,
        ),
    )
    test_run.upload_pair_results(PairResultDataFrame(df_pair_result))

    assert len(test_run.load_remaining_pairs()[1]) == 0

    columns = ["image_pair_id", "similarity"]
    df_pair_result_actual = model.load_pair_results(test_suite)[columns]
    df_pair_result_expected = df_pair_result[columns].groupby("image_pair_id").agg("max").reset_index()
    pandas.testing.assert_frame_equal(df_pair_result_actual, df_pair_result_expected)


@pytest.mark.depends(on=["test__multi_face"])
def test__multi_face__reset(fr_multi_face_test_run: Tuple[Model, TestSuite]) -> None:
    model, test_suite = fr_multi_face_test_run

    reset_model = InferenceModel.load_by_name(
        model.data.name,
        extract=lambda _: None,
        compare=lambda _, __: 0.79,
    )
    test(reset_model, test_suite, reset=True)
    assert_similarity_equal(model.load_pair_results(test_suite), 0.79)


def test__multi_face__invalid(fr_test_suites: List[TestSuite]) -> None:
    name = with_test_prefix(f"{__file__}::test__multi_face__invalid model")
    model = Model.create(name, metadata={})
    test_suite = fr_test_suites[0]

    test_run = TestRun(model, test_suite)

    df_image = test_run.load_remaining_images()
    n_images = len(df_image)
    df_image_result = pd.DataFrame(
        dict(
            image_id=df_image["image_id"].tolist() * 2,
            embedding=[np.random.rand(256).astype(np.float32) for _ in range(n_images)] + ([None] * n_images),
        ),
    )
    with pytest.raises(RemoteError):  # fail to ingest embeddings AND failure to enroll for a given image
        test_run.upload_image_results(ImageResultDataFrame(df_image_result))


def test__upload_image_results__reset(test_samples: List[TestCaseRecord]) -> None:
    prefix = with_test_prefix(f"{__file__}::test__upload_image_results__reset")
    model_0 = InferenceModel.create(
        f"{prefix} model 0",
        extract=lambda _: None,
        compare=lambda _, __: 0.99,
        metadata={},
    )
    test_case_0 = TestCase(
        f"{prefix} test case 0",
        test_samples=test_samples,
        reset=True,
    )
    test_suite_0 = TestSuite(
        f"{prefix} test suite 0",
        baseline_test_cases=[test_case_0],
        reset=True,
    )
    test_run = TestRun(model_0, test_suite_0, reset=True)

    df_remaining_images = test_run.load_remaining_images()
    df_image_results = generate_image_results(df_remaining_images["image_id"].tolist())
    n_uploaded = test_run.upload_image_results(df_image_results)
    assert n_uploaded == len(df_image_results)


def test__upload_pair_results__reset(test_samples: List[TestCaseRecord]) -> None:
    prefix = with_test_prefix(f"{__file__}::test__upload_pair_results__reset")
    model_0 = InferenceModel.create(
        f"{prefix} model 0",
        extract=lambda _: None,
        compare=lambda _, __: 0.99,
        metadata={},
    )
    test_case_0 = TestCase(
        f"{prefix} test case 0",
        test_samples=test_samples,
        reset=True,
    )
    test_suite_0 = TestSuite(
        f"{prefix} test suite 0",
        baseline_test_cases=[test_case_0],
        reset=True,
    )
    test_run = TestRun(model_0, test_suite_0, reset=True)

    df_remaining_images = test_run.load_remaining_images()
    df_image_results = generate_image_results(df_remaining_images["image_id"].tolist())
    test_run.upload_image_results(df_image_results)

    df_embedding, df_image_pair = test_run.load_remaining_pairs()
    df_image_pair_results = generate_pair_results(df_embedding, df_image_pair)
    n_uploaded = test_run.upload_pair_results(df_image_pair_results)
    assert n_uploaded == len(df_image_pair)


def test__test__empty_test_suite() -> None:
    prefix = with_test_prefix(f"{__file__}::test__test__empty_test_suite")
    model_0 = InferenceModel.create(
        f"{prefix} model 0",
        extract=lambda _: None,
        compare=lambda _, __: 0.69,
        metadata={},
    )
    test_suite_0 = TestSuite(
        f"{prefix} test suite 0",
        reset=True,
    )
    with pytest.raises(RemoteError) as exc_info:
        test(model_0, test_suite_0)

    expected_error_msg = "failed to create test run for test suites"
    exc_info_value = str(exc_info.value)
    assert expected_error_msg in exc_info_value
