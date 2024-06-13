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
import kolena.metrics._geometry
from kolena.annotation import BoundingBox
from kolena.annotation import LabeledBoundingBox
from kolena.annotation import ScoredBoundingBox
from kolena.annotation import ScoredLabeledBoundingBox
from kolena.metrics._geometry import InferenceMatches

object_detection = pytest.importorskip("kolena._experimental.object_detection", reason="requires kolena[metrics] extra")


@pytest.mark.metrics
@pytest.mark.parametrize(
    "ground_truths,inferences,expected",
    [
        (
            # ground truths
            [
                [LabeledBoundingBox(label="dog", top_left=[1, 1], bottom_right=[5, 5])],
                [LabeledBoundingBox(label="dog", top_left=[10, 10], bottom_right=[15, 15])],
                [],
                math.nan,
                None,
            ],
            # inferences
            [
                [LabeledBoundingBox(label="cat", top_left=[3, 3], bottom_right=[9, 9])],
                [LabeledBoundingBox(label="dog", top_left=[11, 10], bottom_right=[15, 15])],
                [],
                math.nan,
                None,
            ],
            # expected
            True,
        ),
        (
            # ground truths
            [
                [LabeledBoundingBox(label="dog", top_left=[1, 1], bottom_right=[5, 5])],
                [LabeledBoundingBox(label="dog", top_left=[10, 10], bottom_right=[15, 15])],
                [],
                math.nan,
                None,
            ],
            # inferences
            [
                [LabeledBoundingBox(label="dog", top_left=[3, 3], bottom_right=[9, 9])],
                [LabeledBoundingBox(label="dog", top_left=[11, 10], bottom_right=[15, 15])],
                [],
                math.nan,
                None,
            ],
            # expected
            False,
        ),
        (
            # ground truths
            [
                [LabeledBoundingBox(label="dog", top_left=[1, 1], bottom_right=[5, 5])],
                [BoundingBox(top_left=[10, 10], bottom_right=[15, 15])],
                [],
                math.nan,
                None,
            ],
            # inferences
            [
                [LabeledBoundingBox(label="dog", top_left=[3, 3], bottom_right=[9, 9])],
                [BoundingBox(top_left=[11, 10], bottom_right=[15, 15])],
                [],
                math.nan,
                None,
            ],
            # expected
            False,
        ),
        (
            # ground truths
            [
                [LabeledBoundingBox(label="dog", top_left=[1, 1], bottom_right=[5, 5])],
                [BoundingBox(top_left=[10, 10], bottom_right=[15, 15])],
                [],
                math.nan,
                None,
            ],
            # inferences
            [
                [LabeledBoundingBox(label="cat", top_left=[3, 3], bottom_right=[9, 9])],
                [BoundingBox(top_left=[11, 10], bottom_right=[15, 15])],
                [],
                math.nan,
                None,
            ],
            # expected
            False,
        ),
    ],
)
def test__check_multiclass(ground_truths: list, inferences: list, expected: bool) -> None:
    assert object_detection.dataset._check_multiclass(pd.Series(ground_truths), pd.Series(inferences)) == expected


@pytest.mark.metrics
def test__single_class_datapoint_metrics_adds_thresholded_label_in_single_class_with_label_case() -> None:
    matches = InferenceMatches(
        matched=[
            (
                LabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cat"),
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cat", score=0.6),
            ),
        ],
        unmatched_inf=[ScoredLabeledBoundingBox(top_left=(3, 3), bottom_right=(4, 4), label="cat", score=0.5)],
        unmatched_gt=[LabeledBoundingBox(top_left=(5, 5), bottom_right=(6, 6), label="cat")],
    )
    metrics = object_detection.dataset.single_class_datapoint_metrics(matches, 0.5, [0.2, 0.8])
    assert all(data.get("label", None) == "cat" for data in metrics["thresholded"])


@pytest.mark.metrics
def test__single_class_datapoint_metrics_omits_label_in_single_class_without_label_case() -> None:
    matches = InferenceMatches(
        matched=[
            (
                BoundingBox(top_left=(1, 1), bottom_right=(2, 2)),
                ScoredBoundingBox(top_left=(1, 1), bottom_right=(2, 2), score=0.6),
            ),
        ],
        unmatched_inf=[ScoredBoundingBox(top_left=(3, 3), bottom_right=(4, 4), score=0.5)],
        unmatched_gt=[BoundingBox(top_left=(5, 5), bottom_right=(6, 6))],
    )
    metrics = object_detection.dataset.single_class_datapoint_metrics(matches, 0.5, [0.2, 0.8])
    assert all("label" not in data for data in metrics["thresholded"])


@pytest.mark.metrics
def test__upload_object_detection_ignore_field() -> None:
    locator = "s3://mybucket/image1.jpg"
    ground_truths = [
        BoundingBox(top_left=(0, 0), bottom_right=(1, 1), ignore_flag=True),
        BoundingBox(top_left=(1, 1), bottom_right=(2, 2), ignore_flag=True),
        BoundingBox(top_left=(2, 2), bottom_right=(3, 3), ignore_flag=False),
        BoundingBox(top_left=(3, 3), bottom_right=(4, 4), ignore_flag=False),
        BoundingBox(top_left=(4, 4), bottom_right=(5, 5)),
        BoundingBox(top_left=(5, 5), bottom_right=(6, 6)),
    ]
    with patch.object(
        kolena.dataset,
        "download_dataset",
        return_value=pd.DataFrame([dict(locator=locator, bboxes=ground_truths)]),
    ):
        df = object_detection.dataset.compute_object_detection_results(
            "my dataset",
            pd.DataFrame(
                [
                    dict(
                        locator=locator,
                        predictions=[
                            ScoredBoundingBox(top_left=(0, 0), bottom_right=(1, 1), score=1),
                            ScoredBoundingBox(top_left=(2, 2), bottom_right=(3, 3), score=1),
                            ScoredBoundingBox(top_left=(4, 4), bottom_right=(5, 5), score=1),
                        ],
                    ),
                ],
            ),
            ground_truths_field="bboxes",
            raw_inferences_field="predictions",
            gt_ignore_property="ignore_flag",
            iou_threshold=0.152,
            threshold_strategy="F1-Optimal",
            min_confidence_score=0.222,
        )
    assert [ScoredBoundingBox(**elem) for elem in df["TP"][0]] == [
        ScoredBoundingBox(top_left=(2, 2), bottom_right=(3, 3), score=1, ignore_flag=False),
        ScoredBoundingBox(top_left=(4, 4), bottom_right=(5, 5), score=1),
    ]
    assert df["FN"][0] == [
        BoundingBox(top_left=(3, 3), bottom_right=(4, 4), ignore_flag=False),
        BoundingBox(top_left=(5, 5), bottom_right=(6, 6)),
    ]
    assert df["FP"][0] == []


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
        batch_size=10000,
        ground_truth="bboxes",
        inference="predictions",
        gt_ignore_property=None,
        iou_threshold=0.152,
        threshold_strategy="F1-Optimal",
        min_confidence_score=0.222,
    )
