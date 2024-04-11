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
from typing import Union

import pandas as pd
import pytest

from kolena.annotation import BoundingBox
from kolena.annotation import LabeledBoundingBox
from kolena.annotation import LabeledPolygon
from kolena.annotation import Polygon
from kolena.annotation import ScoredBoundingBox
from kolena.annotation import ScoredLabeledBoundingBox
from kolena.annotation import ScoredLabeledPolygon
from kolena.annotation import ScoredPolygon
from kolena.dataset import download_results
from kolena.dataset import upload_dataset
from tests.integration.helper import fake_locator
from tests.integration.helper import with_test_prefix

object_detection = pytest.importorskip("kolena._experimental.object_detection", reason="requires kolena[metrics] extra")
upload_object_detection_results = object_detection.upload_object_detection_results

ObjectAnnotations = Union[List[BoundingBox], List[Polygon]]

N_DATAPOINTS = 10

gt_labeled_bbox_single = [
    [
        LabeledBoundingBox(label="cat", top_left=[i, i], bottom_right=[i + 10, i + 10], foo="bar"),
        LabeledBoundingBox(label="cat", top_left=[i + 5, i + 5], bottom_right=[i + 20, i + 20], foo="bar"),
    ]
    for i in range(N_DATAPOINTS)
]
inf_labeled_bbox_single = [
    [
        ScoredLabeledBoundingBox(
            label="cat",
            top_left=[box.top_left[0] + 1, box.top_left[1] + 1],
            bottom_right=[box.bottom_right[0] + 3, box.bottom_right[1] + 3],
            score=random.random(),
        )
        for box in boxes
    ]
    for boxes in gt_labeled_bbox_single
]

gt_unlabeled_bbox_single = [
    [
        BoundingBox(top_left=[i, i], bottom_right=[i + 10, i + 10], foo="bar"),
        BoundingBox(top_left=[i + 5, i + 5], bottom_right=[i + 20, i + 20], foo="bar"),
    ]
    for i in range(N_DATAPOINTS)
]
inf_unlabeled_bbox_single = [
    [
        ScoredBoundingBox(
            top_left=[box.top_left[0] + 1, box.top_left[1] + 1],
            bottom_right=[box.bottom_right[0] + 3, box.bottom_right[1] + 3],
            score=random.random(),
        )
        for box in boxes
    ]
    for boxes in gt_labeled_bbox_single
]

gt_labeled_polygon_single = [
    [
        LabeledPolygon(label="cat", points=[(i, i), (i + 10, i), (i, i + 10), (i + 10, i + 10)], foo="bar"),
        LabeledPolygon(
            label="cat",
            points=[(i + 5, i + 5), (i + 20, i + 5), (i + 5, i + 20), (i + 20, i + 20)],
            foo="bar",
        ),
    ]
    for i in range(N_DATAPOINTS)
]
inf_labeled_polygon_single = [
    [
        ScoredLabeledPolygon(label="cat", points=[(x + 1, y + 1) for x, y in polygon.points], score=random.random())
        for polygon in polygons
    ]
    for polygons in gt_labeled_polygon_single
]

gt_unlabeled_polygon_single = [
    [
        Polygon(points=[(i, i), (i + 10, i), (i, i + 10), (i + 10, i + 10)], foo="bar"),
        Polygon(points=[(i + 5, i + 5), (i + 20, i + 5), (i + 5, i + 20), (i + 20, i + 20)], foo="bar"),
    ]
    for i in range(N_DATAPOINTS)
]
inf_unlabeled_polygon_single = [
    [ScoredPolygon(points=[(x + 1, y + 1) for x, y in polygon.points], score=random.random()) for polygon in polygons]
    for polygons in gt_unlabeled_polygon_single
]


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
@pytest.mark.parametrize(
    "annotation,gts,infs",
    [
        ("labeled_bboxes", gt_labeled_bbox_single, inf_labeled_bbox_single),
        ("unlabeled_bboxes", gt_unlabeled_bbox_single, inf_unlabeled_bbox_single),
        ("labeled_polygons", gt_labeled_polygon_single, inf_labeled_polygon_single),
        ("unlabeled_polygons", gt_unlabeled_polygon_single, inf_unlabeled_polygon_single),
    ],
)
def test__upload_results__single_class(annotation: str, gts: ObjectAnnotations, infs: ObjectAnnotations) -> None:
    name = with_test_prefix(f"{__file__}::test__upload_results__{annotation}__single_class")
    datapoints = [
        dict(
            locator=fake_locator(i, name),
            width=i + 500,
            height=i + 400,
            ground_truths=gt,
        )
        for i, gt in enumerate(gts)
    ]
    upload_dataset(name, pd.DataFrame(datapoints))

    inferences = [
        dict(
            locator=dp["locator"],
            raw_inferences=inf,
            theme=random.choice(["animal", "sports", "technology"]),
        )
        for dp, inf in zip(datapoints, infs)
    ]
    eval_config = dict(iou_threshold=0.3, threshold_strategy=0.7, min_confidence_score=0.2)
    upload_object_detection_results(
        name,
        name,
        pd.DataFrame(inferences),
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
        "thresholded",
        "theme",
    }
    assert expected_columns.issubset(set(df_results.columns))
    assert "ground_truths" not in df_results.columns
    assert len(df_results) == 10
    _assert_result_bbox_contains_fields(
        df_results,
        ["TP", "FN", "thresholded"],
        ["foo"],
    )


gt_labeled_bbox_multi = [
    [
        LabeledBoundingBox(label="cat", top_left=[i, i], bottom_right=[i + 30, i + 30], foo="bar"),
        LabeledBoundingBox(label="dog", top_left=[i + 5, i + 5], bottom_right=[i + 50, i + 50], foo="bar"),
        LabeledBoundingBox(label="horse", top_left=[i + 15, i + 15], bottom_right=[i + 60, i + 75], foo="bar"),
    ]
    for i in range(N_DATAPOINTS)
]
inf_score_bbox_multi = [min(0.4 + 0.1 * i, 0.9) for i in range(len(gt_labeled_bbox_multi))]
inf_labeled_bbox_multi = [
    [
        ScoredLabeledBoundingBox(
            label=box.label if i % 4 else "dog",
            top_left=[box.top_left[0] + 1, box.top_left[1] + 1],
            bottom_right=[box.bottom_right[0] + 3, box.bottom_right[1] + 3],
            score=inf_score_bbox_multi[i] + j * 0.01,
        )
        for j, box in enumerate(boxes)
    ]
    for i, boxes in enumerate(gt_labeled_bbox_multi)
]

gt_labeled_polygon_multi = [
    [
        LabeledPolygon(label="cat", points=[(i, i), (i + 30, i), (i, i + 30), (i + 30, i + 30)], foo="bar"),
        LabeledPolygon(
            label="dog",
            points=[(i + 5, i + 5), (i + 50, i + 5), (i + 5, i + 50), (i + 50, i + 50)],
            foo="bar",
        ),
        LabeledPolygon(
            label="horse",
            points=[(i + 15, i + 15), (i + 60, i + 15), (i + 15, i + 75), (i + 60, i + 75)],
            foo="bar",
        ),
    ]
    for i in range(N_DATAPOINTS)
]
inf_score_polygon_multi = [min(0.2 + 0.15 * i, 0.9) for i in range(len(gt_labeled_polygon_multi))]
inf_labeled_polygon_multi = [
    [
        ScoredLabeledPolygon(
            label=polygon.label if i % 4 else "dog",
            points=[(x, y) for x, y in polygon.points],
            score=inf_score_polygon_multi[i] + 0.01 * j,
        )
        for j, polygon in enumerate(polygons)
    ]
    for i, polygons in enumerate(gt_labeled_polygon_multi)
]


@pytest.mark.metrics
@pytest.mark.parametrize(
    "annotation,gts,infs",
    [
        ("labeled_bboxes", gt_labeled_bbox_multi, inf_labeled_bbox_multi),
        ("labeled_polygons", gt_labeled_polygon_multi, inf_labeled_polygon_multi),
    ],
)
def test__upload_results__multiclass(annotation: str, gts: ObjectAnnotations, infs: ObjectAnnotations) -> None:
    name = with_test_prefix(f"{__file__}::test__upload_results__{annotation}__multiclass")
    datapoints = [
        dict(
            locator=fake_locator(i, name),
            width=i + 500,
            height=i + 400,
            ground_truths=gt,
        )
        for i, gt in enumerate(gts)
    ]
    upload_dataset(name, pd.DataFrame(datapoints))

    inferences = [
        dict(
            locator=dp["locator"],
            raw_inferences=inf,
            theme=random.choice(["animal", "sports", "technology"]),
        )
        for dp, inf in zip(datapoints, infs)
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
        **eval_config_one,
    )
    upload_object_detection_results(
        name,
        name,
        pd.DataFrame(inferences),
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
        "raw_inferences",
        "thresholded",
        "theme",
        "Confused",
    }
    assert expected_columns.issubset(set(df_results_one.columns))
    assert "ground_truths" not in df_results_one.columns
    assert len(df_results_one) == 10
    assert len(df_results_two) == 10

    # check data format
    confused = next(x for x in df_results_one["thresholded"])
    assert confused[0].predicted_label
    assert confused[0].predicted_score

    _assert_result_bbox_contains_fields(
        df_results_one,
        ["TP", "FN", "thresholded"],
        ["foo"],
    )
    _assert_result_bbox_contains_fields(
        df_results_two,
        ["TP", "FN", "thresholded"],
        ["foo"],
    )
