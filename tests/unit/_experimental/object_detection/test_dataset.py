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
import math
from unittest.mock import Mock
from unittest.mock import patch

import pandas as pd
import pytest

import kolena.dataset
from kolena.annotation import LabeledBoundingBox

object_detection = pytest.importorskip("kolena._experimental.object_detection", reason="requires kolena[metrics] extra")


@pytest.mark.metrics
def test__check_multiclass() -> None:
    assert (
        object_detection.dataset._check_multiclass(
            pd.Series(
                [
                    [LabeledBoundingBox(label="dog", top_left=[1, 1], bottom_right=[5, 5])],
                    [LabeledBoundingBox(label="dog", top_left=[10, 10], bottom_right=[15, 15])],
                    [],
                    math.nan,
                    None,
                ],
            ),
            pd.Series(
                [
                    [LabeledBoundingBox(label="cat", top_left=[3, 3], bottom_right=[9, 9])],
                    [LabeledBoundingBox(label="dog", top_left=[11, 10], bottom_right=[15, 15])],
                    [],
                    math.nan,
                    None,
                ],
            ),
        )
        is True
    )

    assert (
        object_detection.dataset._check_multiclass(
            pd.Series(
                [
                    [LabeledBoundingBox(label="dog", top_left=[1, 1], bottom_right=[5, 5])],
                    [LabeledBoundingBox(label="dog", top_left=[10, 10], bottom_right=[15, 15])],
                    [],
                    math.nan,
                    None,
                ],
            ),
            pd.Series(
                [
                    [LabeledBoundingBox(label="dog", top_left=[3, 3], bottom_right=[9, 9])],
                    [LabeledBoundingBox(label="dog", top_left=[11, 10], bottom_right=[15, 15])],
                    [],
                    math.nan,
                    None,
                ],
            ),
        )
        is False
    )


@pytest.mark.metrics
@patch("kolena.dataset.upload_results")
def test__upload_object_detection_results_configurations(mocked_upload_results: Mock) -> None:
    locator = "s3://mybucket/image1.jpg"
    with patch.object(object_detection.dataset, "_compute_metrics") as patched_metrics:
        with patch.object(
            kolena.dataset,
            "download_dataset",
            return_value=pd.DataFrame([dict(locator=locator, bboxes=[])]),
        ):
            object_detection.dataset.upload_object_detection_results(
                "my dataset",
                "my model",
                pd.DataFrame([dict(locator=locator, predictions=[])]),
                ground_truths_field="bboxes",
                raw_inferences_field="predictions",
                iou_threshold=0.152,
                threshold_strategy="F1-Optimal",
                min_confidence_score=0.222,
            )

    patched_metrics.assert_called_once()
    _, kwargs = patched_metrics.call_args
    assert kwargs == dict(
        ground_truth="bboxes",
        inference="predictions",
        iou_threshold=0.152,
        threshold_strategy="F1-Optimal",
        min_confidence_score=0.222,
    )
