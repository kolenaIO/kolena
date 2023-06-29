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
from kolena._experimental.object_detection import Inference
from kolena.workflow.annotation import ScoredLabeledBoundingBox


def test__inference__empty() -> None:
    test_bboxes = []
    inf = Inference(
        bboxes=test_bboxes,
    )
    assert inf.bboxes == test_bboxes
    assert inf._to_dict() == dict([("bboxes", [])])


def test__inference__simple() -> None:
    box1 = ScoredLabeledBoundingBox(top_left=(0, 0), bottom_right=(1.1, 1.1), label="test_label", score=0.5)
    test_bboxes = [box1]
    inf = Inference(
        bboxes=test_bboxes,
    )
    assert inf.bboxes == test_bboxes

    assert inf._to_dict() == dict(
        [
            (
                "bboxes",
                [
                    dict(
                        [
                            ("top_left", [0.0, 0.0]),
                            ("bottom_right", [1.1, 1.1]),
                            ("label", "test_label"),
                            ("score", 0.5),
                            ("data_type", "ANNOTATION/BOUNDING_BOX"),
                        ],
                    ),
                ],
            ),
        ],
    )


def test__inference__advanced() -> None:
    box1 = ScoredLabeledBoundingBox(top_left=(0, 0), bottom_right=(1.1, 1.1), label="test_label", score=0.5)
    box2 = ScoredLabeledBoundingBox(top_left=(0, 0), bottom_right=(1.1, 1.1), label="a", score=0)
    box3 = ScoredLabeledBoundingBox(top_left=(0, 0), bottom_right=(1.1, 1.1), label="b", score=1)
    test_bboxes = [box1, box2, box3]
    inf = Inference(
        bboxes=test_bboxes,
    )
    assert inf.bboxes == test_bboxes

    assert inf._to_dict() == dict(
        [
            (
                "bboxes",
                [
                    dict(
                        [
                            ("top_left", [0.0, 0.0]),
                            ("bottom_right", [1.1, 1.1]),
                            ("label", "test_label"),
                            ("score", 0.5),
                            ("data_type", "ANNOTATION/BOUNDING_BOX"),
                        ],
                    ),
                    dict(
                        [
                            ("top_left", [0.0, 0.0]),
                            ("bottom_right", [1.1, 1.1]),
                            ("label", "a"),
                            ("score", 0),
                            ("data_type", "ANNOTATION/BOUNDING_BOX"),
                        ],
                    ),
                    dict(
                        [
                            ("top_left", [0.0, 0.0]),
                            ("bottom_right", [1.1, 1.1]),
                            ("label", "b"),
                            ("score", 1),
                            ("data_type", "ANNOTATION/BOUNDING_BOX"),
                        ],
                    ),
                ],
            ),
        ],
    )
