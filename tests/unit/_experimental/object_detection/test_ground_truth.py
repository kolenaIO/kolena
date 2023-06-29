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
from kolena._experimental.object_detection import GroundTruth
from kolena.workflow.annotation import LabeledBoundingBox


def test__ground__truth__empty() -> None:
    test_bboxes = []
    gt = GroundTruth(
        bboxes=test_bboxes,
    )
    assert gt.bboxes == test_bboxes
    assert gt.ignored_bboxes == []
    assert gt.labels == []
    assert gt.n_bboxes == 0

    assert gt._to_dict() == dict(
        [
            ("bboxes", []),
            ("ignored_bboxes", []),
            ("labels", []),
            ("n_bboxes", 0),
        ],
    )


def test__ground__truth__simple() -> None:
    box1 = LabeledBoundingBox(top_left=(0, 0), bottom_right=(1.1, 1.1), label="test_label")
    test_bboxes = [box1]
    gt = GroundTruth(
        bboxes=test_bboxes,
    )
    assert gt.bboxes == test_bboxes
    assert gt.ignored_bboxes == []
    assert gt.labels == ["test_label"]
    assert gt.n_bboxes == 1

    assert gt._to_dict() == dict(
        [
            (
                "bboxes",
                [
                    dict(
                        [
                            ("top_left", [0.0, 0.0]),
                            ("bottom_right", [1.1, 1.1]),
                            ("label", "test_label"),
                            ("data_type", "ANNOTATION/BOUNDING_BOX"),
                        ],
                    ),
                ],
            ),
            ("ignored_bboxes", []),
            ("labels", ["test_label"]),
            ("n_bboxes", 1),
        ],
    )


def test__ground__truth__advanced() -> None:
    box1 = LabeledBoundingBox(top_left=(0, 0), bottom_right=(1.1, 1.1), label="test_label")
    box2 = LabeledBoundingBox(top_left=(0, 0), bottom_right=(1.1, 1.1), label="a")
    box3 = LabeledBoundingBox(top_left=(0, 0), bottom_right=(1.1, 1.1), label="b")
    test_bboxes = [box1, box2]
    test_ignored_bboxes = [box1, box3]
    gt = GroundTruth(
        bboxes=test_bboxes,
        ignored_bboxes=test_ignored_bboxes,
    )
    assert gt.bboxes == test_bboxes
    assert gt.ignored_bboxes == test_ignored_bboxes
    assert gt.labels == ["a", "test_label"]
    assert gt.n_bboxes == 2

    assert gt._to_dict() == dict(
        [
            (
                "bboxes",
                [
                    dict(
                        [
                            ("top_left", [0.0, 0.0]),
                            ("bottom_right", [1.1, 1.1]),
                            ("label", "test_label"),
                            ("data_type", "ANNOTATION/BOUNDING_BOX"),
                        ],
                    ),
                    dict(
                        [
                            ("top_left", [0.0, 0.0]),
                            ("bottom_right", [1.1, 1.1]),
                            ("label", "a"),
                            ("data_type", "ANNOTATION/BOUNDING_BOX"),
                        ],
                    ),
                ],
            ),
            (
                "ignored_bboxes",
                [
                    dict(
                        [
                            ("top_left", [0.0, 0.0]),
                            ("bottom_right", [1.1, 1.1]),
                            ("label", "test_label"),
                            ("data_type", "ANNOTATION/BOUNDING_BOX"),
                        ],
                    ),
                    dict(
                        [
                            ("top_left", [0.0, 0.0]),
                            ("bottom_right", [1.1, 1.1]),
                            ("label", "b"),
                            ("data_type", "ANNOTATION/BOUNDING_BOX"),
                        ],
                    ),
                ],
            ),
            ("labels", ["a", "test_label"]),
            ("n_bboxes", 2),
        ],
    )
