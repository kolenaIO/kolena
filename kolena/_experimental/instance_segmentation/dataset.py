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
from kolena._experimental.object_detection.dataset import upload_object_detection_results


upload_instance_segmentation_results = upload_object_detection_results
"""
Compute metrics and upload results of the model for the dataset.

Dataframe `df` should include a `locator` column that would match to that of corresponding datapoint. Column
:inference in the Dataframe `df` should be a list of scored [`Polygons`][kolena.annotation.Polygon].

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
