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
from typing import Dict
from typing import Literal
from typing import Union

import pandas as pd

from kolena._experimental.object_detection.dataset import upload_object_detection_results


def upload_instance_segmentation_results(
    dataset_name: str,
    model_name: str,
    df: pd.DataFrame,
    *,
    ground_truths_field: str = "ground_truths",
    raw_inferences_field: str = "raw_inferences",
    iou_threshold: float = 0.5,
    threshold_strategy: Union[Literal["F1-Optimal"], float, Dict[str, float]] = "F1-Optimal",
    min_confidence_score: float = 0.01,
    batch_size: int = 10_000,
) -> None:
    """
    Compute metrics and upload results of the model for the dataset.

    Dataframe `df` should include a `locator` column that would match to that of corresponding datapoint and
    an `inference` column that should be a list of scored [`Polygons`][kolena.annotation.Polygon].

    :param dataset_name: Dataset name.
    :param model_name: Model name.
    :param df: Dataframe for model results.
    :param ground_truths_field: Field name in datapoint with ground truth polygons, defaulting to `"ground_truths"`.
    :param raw_inferences_field: Column in model result DataFrame with raw inference polygons,
    defaulting to `"raw_inferences"`.
    :param iou_threshold: The [IoU ↗](../../metrics/iou.md) threshold, defaulting to `0.5`.
    :param threshold_strategy: The confidence threshold strategy. It can either be a fixed confidence threshold such
        as `0.5` or `0.75`, or `"F1-Optimal"` to find the threshold maximizing F1 score..
    :param min_confidence_score: The minimum confidence score to consider for the evaluation. This is usually set to
        reduce noise by excluding inferences with low confidence score.
    :return:
    """
    return upload_object_detection_results(
        dataset_name,
        model_name,
        df,
        ground_truths_field=ground_truths_field,
        raw_inferences_field=raw_inferences_field,
        iou_threshold=iou_threshold,
        threshold_strategy=threshold_strategy,
        min_confidence_score=min_confidence_score,
        batch_size=batch_size,
    )
