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
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from requests import HTTPError

from kolena.fr import InferenceModel
from kolena.fr import Model
from kolena.fr import TestCase
from kolena.fr import TestRun
from kolena.fr import TestSuite
from tests.integration.fr.helper import generate_image_results
from tests.integration.helper import with_test_prefix


def test__create() -> None:
    model_name = with_test_prefix(f"{__file__} test_create")
    model_metadata = {
        "detector": "det",
        "landmarks_estimator": "lmks",
        "quality_predictor": "qual",
        "embedding_extractor": "emb",
    }

    created_model = Model.create(name=model_name, metadata=model_metadata)
    assert created_model.data.name == model_name
    assert created_model.data.metadata == model_metadata


def test__create__validate_name() -> None:
    with pytest.raises(ValueError):
        Model.create(name=" ", metadata={})


def test__load_by_name() -> None:
    model_name = with_test_prefix(f"{__file__} test_load_by_name")
    created_model = Model.create(name=model_name, metadata={})
    loaded_model = Model.load_by_name(model_name)
    assert created_model == loaded_model


def test__load_by_name__nonexistent() -> None:
    model_name = with_test_prefix(f"{__file__} test_load_by_name_nonexistent")
    # TODO: Should not be HTTPError
    with pytest.raises(HTTPError):
        Model.load_by_name(model_name)


def test__create__bad_metadata() -> None:
    model_name = with_test_prefix(f"{__file__} test_create_bad_metadata")
    with pytest.raises(ValidationError):
        Model.create(name=model_name, metadata=cast(Dict[str, Any], None))
    with pytest.raises(ValidationError):
        Model.create(name=model_name, metadata=cast(Dict[str, Any], "bogus"))


def test__create__existing() -> None:
    model_name = with_test_prefix(f"{__file__} test_create_existing")
    Model.create(name=model_name, metadata={})
    # TODO: Should not be HTTPError
    with pytest.raises(HTTPError):
        Model.create(name=model_name, metadata={})


def test__load_by_name__seeded(fr_models: List[Model]) -> None:
    for i, model in enumerate(fr_models):
        db_model = Model.load_by_name(model.data.name)
        assert db_model.data == model.data
        inference_model = InferenceModel.load_by_name(
            model.data.name,
            extract=lambda _: None,
            compare=lambda a, b: len(a) + len(b),
        )
        assert inference_model.data == model.data
        assert inference_model.extract("unimportant") is None
        assert inference_model.compare(np.ones(1), np.ones(2)) == 3


def test__load_pair_results__empty(fr_test_cases: List[TestCase]) -> None:
    name = with_test_prefix(f"{__file__} test_load_pair_results_empty")
    model = Model.create(name, dict(test="metadata"))

    # no results for model on test suite/case should not throw
    test_case_record = fr_test_cases[0]
    test_case = TestCase.load_by_name(test_case_record.name, version=test_case_record.version)
    df = model.load_pair_results(test_case)
    assert len(df) == 0

    # empty test suite should not throw
    test_suite = TestSuite.create(with_test_prefix(f"{__file__} test_load_pair_results_empty test suite"))
    df = model.load_pair_results(test_suite)
    assert len(df) == 0


def test__iter_pair_results__empty(fr_test_cases: List[TestCase]) -> None:
    name = with_test_prefix(f"{__file__} test_iter_pair_results_empty")
    model = Model.create(name, dict(test="metadata"))

    # no results for model on test suite/case should not throw
    test_case_record = fr_test_cases[0]
    test_case = TestCase.load_by_name(test_case_record.name, version=test_case_record.version)
    for _ in model.iter_pair_results(test_case):
        pytest.fail("expected no data to iterate over")

    # empty test suite should not throw
    test_suite = TestSuite.create(with_test_prefix(f"{__file__} test_iter_pair_results_empty test suite"))
    for _ in model.iter_pair_results(test_suite):
        pytest.fail("expected no data to iterate over")


def _seed_results(model_name: str, test_suite: TestSuite) -> Tuple[Model, pd.DataFrame, Set[str]]:
    model = Model.create(model_name, dict(example="metadata"))
    test_run = TestRun(model, test_suite)

    df_image = test_run.load_remaining_images()
    fte_image_ids = set(df_image["image_id"].tolist()[:2])
    fte_locators = {record.locator for record in df_image.itertuples() if record.image_id in fte_image_ids}
    df_image_result = generate_image_results(df_image["image_id"].tolist())
    df_image_result["embedding"].mask(df_image_result["image_id"].isin(fte_image_ids), None, inplace=True)
    test_run.upload_image_results(df_image_result)

    df_embedding, df_pair = test_run.load_remaining_pairs()
    df_pair["similarity"] = [
        np.random.rand(1)[0].astype(np.float32)
        if record.image_a_id not in fte_image_ids and record.image_b_id not in fte_image_ids
        else None
        for record in df_pair.itertuples()
    ]
    subset_columns = ["image_pair_id", "similarity"]
    df_pair_result = df_pair[subset_columns]
    test_run.upload_pair_results(df_pair_result)

    df_expected = df_pair_result.sort_values(by=["image_pair_id"])
    return model, df_expected, fte_locators


def test__load_pair_results(fr_test_suites: List[TestSuite]) -> None:
    test_suite = fr_test_suites[0]
    model_name = with_test_prefix(f"{__file__} test_load_pair_results")
    model, df_expected, fte_locators = _seed_results(model_name, test_suite)
    # TODO: ideally would also verify locator_a, locator_b, and is_same but that is a pain to manually splice
    #  together from the frames in fr_test_data
    df_loaded = model.load_pair_results(test_suite).sort_values(by=["image_pair_id"])
    assert np.allclose(df_loaded["similarity"], df_expected["similarity"], equal_nan=True)

    assert df_loaded["image_a_fte"].tolist() == [record.locator_a in fte_locators for record in df_loaded.itertuples()]
    assert df_loaded["image_b_fte"].tolist() == [record.locator_b in fte_locators for record in df_loaded.itertuples()]
    assert all(
        np.isnan(record.similarity)
        for record in df_loaded.itertuples()
        if record.locator_a in fte_locators or record.locator_b in fte_locators
    )


def test__iter_pair_results(fr_test_suites: List[TestSuite]) -> None:
    test_suite = fr_test_suites[0]
    model_name = with_test_prefix(f"{__file__} test_iter_pair_results")
    model, df_expected, _ = _seed_results(model_name, test_suite)
    frames: List[pd.DataFrame] = []
    for df in model.iter_pair_results(test_object=test_suite, batch_size=2):
        frames.append(df)
    df_loaded = pd.concat(frames).sort_values(by=["image_pair_id"])
    assert np.allclose(df_loaded["similarity"], df_expected["similarity"], equal_nan=True)
