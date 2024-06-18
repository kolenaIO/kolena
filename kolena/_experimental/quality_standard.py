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
from typing import Union

import pandas as pd

from kolena._api.v1.event import EventAPI
from kolena._api.v2.quality_standard import Path
from kolena._utils import krequests_v2 as krequests
from kolena._utils import log
from kolena._utils.instrumentation import with_event


def _format_quality_standard_result_df(quality_standard_result: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(quality_standard_result).rename_axis("stratification")

    df = df.explode("results_by_stratification")
    df = pd.concat(
        [df.drop(["results_by_stratification"], axis=1), df["results_by_stratification"].apply(pd.Series)],
        axis=1,
    )
    df.set_index(["test_case"], append=True, drop=True, inplace=True)

    df = df.explode("model_results")
    df["model"] = df["model_results"].apply(lambda x: x["model"]["name"])
    df["eval_config"] = df["model_results"].apply(lambda x: json.dumps(x["model"]["eval_config"]))
    df["metric_group"] = df["model_results"].apply(lambda x: list(x["results"].keys()))

    df = df.explode("metric_group")
    df["metric"] = df.apply(lambda x: list(x["model_results"]["results"][x["metric_group"]].keys()), axis=1)

    df = df.explode("metric")
    df["value"] = df.apply(lambda x: x["model_results"]["results"][x["metric_group"]][x["metric"]], axis=1)

    return df.pivot(columns=["model", "eval_config", "metric_group", "metric"], values="value")


@with_event(event_name=EventAPI.Event.FETCH_QUALITY_STANDARD_RESULT)
def download_quality_standard_result(
    dataset: str,
    models: List[str],
    metric_groups: Union[List[str], None] = None,
) -> pd.DataFrame:
    """
    Download quality standard result given a dataset and list of models.

    :param dataset: The name of the dataset.
    :param models: The names of the models.
    :param metric_groups: The names of the metric groups to include in the result.
    Note all metric groups are included when this value is `None`.
    :return: A Dataframe containing the quality standard result.
    """
    model_log = ", ".join([f"'{model}'" for model in models])
    log.info(f"downloading quality standard results for model(s) {model_log} on dataset '{dataset}'")
    response = krequests.get(
        Path.RESULT,
        params=dict(dataset_name=dataset, models=models, metric_groups=metric_groups),
        api_version="v2",
    )
    krequests.raise_for_status(response)
    return _format_quality_standard_result_df(response.json())
