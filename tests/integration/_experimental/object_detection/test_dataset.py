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
import random
from typing import List

import pandas as pd
import pytest

from kolena.annotation import LabeledBoundingBox
from kolena.annotation import ScoredLabeledBoundingBox
from kolena.dataset import download_results
from kolena.dataset import upload_dataset
from tests.integration.helper import fake_locator
from tests.integration.helper import with_test_prefix

object_detection = pytest.importorskip("kolena._experimental.object_detection", reason="requires kolena[metrics] extra")
upload_object_detection_results = object_detection.upload_object_detection_results


def _assert_result_bbox_contains_fields(df_results: pd.DataFrame, columns: List[str], fields: List[str]):
    """
    Asserts that each bounding box under the specified columns contain fields.
    """
    for col in columns:
        for result_list in df_results[col]:
            for result in result_list:
                for field in fields:
                    assert field in result.toDict()


@pytest.mark.metrics
def test__upload_results__single_class() -> None:
    name = with_test_prefix(f"{__file__}::test__upload_results__single_class")
    datapoints = [
        dict(
            locator=fake_locator(i, name),
            width=i + 500,
            height=i + 400,
            bboxes=[
                LabeledBoundingBox(label="cat", top_left=[i, i], bottom_right=[i + 10, i + 10], flag="T", foo="bar"),
                LabeledBoundingBox(
                    label="cat",
                    top_left=[i + 5, i + 5],
                    bottom_right=[i + 20, i + 20],
                    flag="F",
                    foo="bar2",
                ),
            ],
        )
        for i in range(10)
    ]
    upload_dataset(name, pd.DataFrame(datapoints))

    inferences = [
        dict(
            locator=dp["locator"],
            raw_inferences=[
                ScoredLabeledBoundingBox(
                    label="cat",
                    top_left=[box.top_left[0] + 1, box.top_left[1] + 1],
                    bottom_right=[box.bottom_right[0] + 3, box.bottom_right[1] + 3],
                    score=random.random(),
                )
                for box in dp["bboxes"]
            ],
            theme=random.choice(["animal", "sports", "technology"]),
        )
        for dp in datapoints
    ]
    eval_config = dict(iou_threshold=0.3, threshold_strategy=0.7, min_confidence_score=0.2)
    upload_object_detection_results(
        name,
        name,
        pd.DataFrame(inferences),
        ground_truths_field="bboxes",
        raw_inferences_field="raw_inferences",
        iou_threshold=0.3,
        threshold_strategy=0.7,
        min_confidence_score=0.2,
    )

    _, results = download_results(name, name)
    assert len(results) == 1
    assert results[0][0] == eval_config

    df_results = results[0][1]
    expected_columns = {
        "TP",
        "FP",
        "FN",
        "raw_inferences",
        "matched_inference",
        "unmatched_inference",
        "unmatched_ground_truth",
        "theme",
    }
    assert expected_columns.issubset(set(df_results.columns))
    assert "bboxes" not in df_results.columns
    assert len(df_results) == 10
    _assert_result_bbox_contains_fields(
        df_results,
        ["TP", "FN", "matched_inference", "unmatched_ground_truth"],
        ["flag", "foo"],
    )


@pytest.mark.metrics
def test__upload_results__multiclass() -> None:
    name = with_test_prefix(f"{__file__}::test__upload_results__multiclass")
    datapoints = [
        dict(
            locator=fake_locator(i, name),
            width=i + 500,
            height=i + 400,
            bounding_boxes=[
                LabeledBoundingBox(label="cat", top_left=[i, i], bottom_right=[i + 30, i + 30], foo="bar1"),
                LabeledBoundingBox(label="dog", top_left=[i + 5, i + 5], bottom_right=[i + 50, i + 50], foo="bar2"),
                LabeledBoundingBox(label="horse", top_left=[i + 15, i + 25], bottom_right=[i + 60, i + 75], foo="bar3"),
            ],
        )
        for i in range(10)
    ]
    upload_dataset(name, pd.DataFrame(datapoints))

    inferences = [
        dict(
            locator=dp["locator"],
            inferences=[
                ScoredLabeledBoundingBox(
                    label=box.label if i % 4 else "dog",
                    top_left=[box.top_left[0] + 1, box.top_left[1] + 1],
                    bottom_right=[box.bottom_right[0] + 3, box.bottom_right[1] + 3],
                    score=0.7,
                )
                for box in dp["bounding_boxes"]
            ],
            theme=random.choice(["animal", "sports", "technology"]),
        )
        for i, dp in enumerate(datapoints)
    ]
    eval_config_one = dict(
        iou_threshold=0.3,
        threshold_strategy=dict(cat=0.7, dog=0.5, horse=0.3),
        min_confidence_score=0.2,
    )
    eval_config_two = dict(iou_threshold=0.5, threshold_strategy="F1-Optimal", min_confidence_score=0.3)
    upload_object_detection_results(
        name,
        name,
        pd.DataFrame(inferences),
        ground_truths_field="bounding_boxes",
        raw_inferences_field="inferences",
        **eval_config_one,
    )
    upload_object_detection_results(
        name,
        name,
        pd.DataFrame(inferences),
        ground_truths_field="bounding_boxes",
        raw_inferences_field="inferences",
        **eval_config_two,
    )

    _, results = download_results(name, name)
    assert len(results) == 2
    assert results[0][0] == eval_config_one
    assert results[1][0] == eval_config_two

    df_results_one = results[0][1]
    df_results_two = results[1][1]
    expected_columns = {
        "TP",
        "FP",
        "FN",
        "inferences",
        "matched_inference",
        "unmatched_inference",
        "unmatched_ground_truth",
        "theme",
        "Confused",
    }
    assert expected_columns.issubset(set(df_results_one.columns))
    assert "bounding_boxes" not in df_results_one.columns
    assert len(df_results_one) == 10
    assert len(df_results_two) == 10

    # check data format
    confused = next(x for x in df_results_one["unmatched_ground_truth"] if len(x))
    assert confused[0].predicted_label
    assert confused[0].predicted_score

    _assert_result_bbox_contains_fields(
        df_results_one,
        ["TP", "FN", "matched_inference", "unmatched_ground_truth"],
        ["foo"],
    )
    _assert_result_bbox_contains_fields(
        df_results_two,
        ["TP", "FN", "matched_inference", "unmatched_ground_truth"],
        ["foo"],
    )
