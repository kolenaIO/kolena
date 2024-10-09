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
import pytest

from kolena._experimental.dataset.evaluation import download_results_by_tag
from kolena.dataset import EvalConfigResults
from kolena.dataset import upload_dataset
from kolena.dataset.evaluation import _upload_results
from kolena.dataset.evaluation import get_models
from kolena.errors import IncorrectUsageError
from kolena.errors import NotFoundError
from tests.integration.dataset.test_evaluation import check_eval_config_result_tuples
from tests.integration.dataset.test_evaluation import get_df_dp
from tests.integration.dataset.test_evaluation import get_df_result
from tests.integration.dataset.test_evaluation import ID_FIELDS
from tests.integration.dataset.test_evaluation import JOIN_COLUMN
from tests.integration.helper import assert_frame_equal
from tests.integration.helper import with_test_prefix


def test__download_results_by_tag__model_does_not_exist() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__download_results_by_tag__model_does_not_exist")
    model_tag = with_test_prefix(f"{__file__}::test__download_results_by_tag__model_does_not_exist")

    with pytest.raises(NotFoundError) as exc_info:
        download_results_by_tag(dataset_name, model_tag)
    exc_info_value = str(exc_info.value)
    assert "no models with tag" in exc_info_value


def test__download_results_by_tag() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__download_results_by_tag")
    model_name = with_test_prefix(f"{__file__}::test__download_results_by_tag")
    model_tags = [
        with_test_prefix(f"{__file__}::test__download_results_by_tag-a"),
        with_test_prefix(f"{__file__}::test__download_results_by_tag-b"),
    ]

    df_dp = get_df_dp()
    dp_columns = [JOIN_COLUMN, "locator", "width", "height", "city"]
    upload_dataset(dataset_name, df_dp[dp_columns], id_fields=ID_FIELDS)

    df_result = get_df_result(10)
    eval_config = dict(threshold=0.422)

    response = _upload_results(
        dataset_name,
        model_name,
        [EvalConfigResults(eval_config, df_result)],
        tags=model_tags,
    )
    assert response.n_inserted == 10
    assert response.n_updated == 0
    assert response.model_id is not None
    assert response.eval_config_id is not None

    models = get_models(dataset_name)
    assert len(models) == 1
    assert models[0].name == model_name
    assert sorted(models[0].tags) == sorted(model_tags)

    for tag in model_tags:
        fetched_df_dp, df_results_by_eval = download_results_by_tag(dataset_name, tag)
        check_eval_config_result_tuples(df_results_by_eval)
        eval_cfg, fetched_df_result = df_results_by_eval[0]
        assert not fetched_df_dp.empty
        assert not fetched_df_result.empty
        assert len(df_results_by_eval) == 1
        assert eval_cfg == eval_config
        assert_frame_equal(fetched_df_dp, fetched_df_result, ID_FIELDS)


def test__download_results_by_tag__multiple_model_with_same_tag() -> None:
    dataset_name = with_test_prefix(f"{__file__}::test__multiple_model_with_same_tag")
    model_name_1 = with_test_prefix(f"{__file__}::test__multiple_model_with_same_tag_1")
    model_name_2 = with_test_prefix(f"{__file__}::test__multiple_model_with_same_tag_2")
    model_tag = with_test_prefix(f"{__file__}::test__multiple_model_with_same_tag")

    df_dp = get_df_dp()
    dp_columns = [JOIN_COLUMN, "locator", "width", "height", "city"]
    upload_dataset(dataset_name, df_dp[dp_columns], id_fields=ID_FIELDS)

    df_result = get_df_result(10)
    eval_config = dict(threshold=0.422)

    # with only 1 model associated with the tag
    response = _upload_results(
        dataset_name,
        model_name_1,
        [EvalConfigResults(eval_config, df_result)],
        tags=[model_tag],
    )
    assert response.n_inserted == 10
    assert response.n_updated == 0
    assert response.model_id is not None
    assert response.eval_config_id is not None
    model_id_1 = response.model_id

    fetched_df_dp, df_results_by_eval = download_results_by_tag(dataset_name, model_tag)
    check_eval_config_result_tuples(df_results_by_eval)
    eval_cfg, fetched_df_result = df_results_by_eval[0]
    assert not fetched_df_dp.empty
    assert not fetched_df_result.empty
    assert len(df_results_by_eval) == 1
    assert eval_cfg == eval_config
    assert_frame_equal(fetched_df_dp, fetched_df_result, ID_FIELDS)

    # with 2 models having the same tag
    response = _upload_results(
        dataset_name,
        model_name_2,
        [EvalConfigResults(eval_config, df_result)],
        tags=[model_tag],
    )
    assert response.n_inserted == 10
    assert response.n_updated == 0
    assert response.model_id is not None
    assert response.eval_config_id is not None
    assert response.model_id != model_id_1

    # cannot download model results by tag if the tag is associated with multiple models
    with pytest.raises(IncorrectUsageError) as exc_info:
        download_results_by_tag(dataset_name, model_tag)
    exc_info_value = str(exc_info.value)
    assert "multiple models with tag" in exc_info_value
    assert model_name_1 in exc_info_value
    assert model_name_2 in exc_info_value

    # can get all models on the dataset as well as their tags
    models = get_models(dataset_name)
    assert len(models) == 2
    models.sort(key=lambda model: model.name)
    assert models[0].name == model_name_1
    assert sorted(models[0].tags) == [model_tag]
    assert models[1].name == model_name_2
    assert sorted(models[1].tags) == [model_tag]
