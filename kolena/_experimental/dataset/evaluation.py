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
from dataclasses import asdict
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd

from kolena._api.v1.event import EventAPI
from kolena._api.v2.model import LoadByTagRequest
from kolena._api.v2.model import LoadByTagResponse
from kolena._api.v2.model import Path
from kolena._utils import krequests_v2 as krequests
from kolena._utils.instrumentation import with_event
from kolena._utils.serde import from_dict
from kolena.dataset.evaluation import download_results
from kolena.dataset.evaluation import EvalConfigResults
from kolena.errors import IncorrectUsageError
from kolena.errors import NotFoundError


def _get_models_by_tag(tag: str) -> List[str]:
    """
    Get model names by a given tag.

    :param tag: The tag on the model.
    :return: A list of model names associated with the provided tag.
    """
    response = krequests.put(
        Path.LOAD_BY_TAG,
        json=asdict(LoadByTagRequest(tag=tag)),
    )
    response.raise_for_status()

    load_by_tag_response = from_dict(LoadByTagResponse, response.json())
    return [model.name for model in load_by_tag_response.models]


@with_event(EventAPI.Event.FETCH_DATASET_MODEL_RESULT_BY_TAG)
def download_results_by_tag(
    dataset: str,
    model_tag: str,
    commit: Optional[str] = None,
    include_extracted_properties: bool = False,
) -> Tuple[pd.DataFrame, List[EvalConfigResults]]:
    """
    Download results given dataset name and model tag. Currently restricted to if the model tag is unique to the model.

    Concat dataset with results:

    ```python
    df_dp, results = download_results_by_tag("dataset name", "model tag")
    for eval_config, df_result in results:
        df_combined = pd.concat([df_dp, df_result], axis=1)
    ```

    :param dataset: The name of the dataset.
    :param model_tag: The tag associated with the model.
    :param commit: The commit hash for version control. Get the latest commit when this value is `None`.
    :param include_extracted_properties: If True, include kolena extracted properties from automated extractions
    in the datapoints and results as separate columns
    :return: Tuple of DataFrame of datapoints and list of [`EvalConfigResults`][kolena.dataset.EvalConfigResults].
    """
    model_names = _get_models_by_tag(model_tag)
    if not model_names:
        raise NotFoundError(f"no models with tag '{model_tag}' were found")
    elif len(model_names) > 1:
        raise IncorrectUsageError(f"multiple models with tag '{model_tag}' were found: {model_names}")
    else:
        return download_results(dataset, model_names[0], commit, include_extracted_properties)
