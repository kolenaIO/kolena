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
import pandas as pd
import pytest

from kolena.annotation import LabeledTextSegment
from kolena.utils import extract_labeled_text_segments_from_keywords


@pytest.mark.parametrize(
    "text,keywords,expected_segments",
    [
        (
            "Kolena is a comprehensive machine learning testing and debugging platform to surface hidden"
            " model behaviors and take the mystery out of model development.",
            ["Kolena", "machine learning", "model development"],
            [(0, 6), (26, 42), (136, 153)],
        ),
        ("The king himself told the audience that he is him", ["queen", "him"], [(46, 49)]),
        (
            "Creating fine-grained tests is labor-intensive and typically involves manual annotation of countless "
            "images, a costly and time-consuming process",
            ["fine-grained tests", "manual annotation", "and"],
            [(9, 27), (70, 87), (47, 50), (118, 121)],
        ),
    ],
)
def test__extract_labeled_text_segments_from_keywords(
    text: str,
    keywords: list[str],
    expected_segments: list[tuple[int, int]],
) -> None:
    df = pd.DataFrame({"text": [text]})
    labeled_text_segments = extract_labeled_text_segments_from_keywords(
        df,
        ["text"],
        {"test_label": keywords},
    )
    actual_segments_row = labeled_text_segments["labeled_text_segments"].iloc[0]
    expected = [
        LabeledTextSegment(
            text_field="text",
            start=start,
            end=end,
            label="test_label",
        )
        for start, end in expected_segments
    ]
    assert actual_segments_row == expected
    for segment in actual_segments_row:
        assert text[segment.start : segment.end] in keywords


def test__extract_labeled_text_segments_from_keywords__multi_label_and_field() -> None:
    text1 = "Perform high-resolution model evaluation"
    text2 = "Understand and track behavioral improvements and regressions"
    text3 = "Meaningfully communicate model capabilities"
    text4 = "Automate model testing and deployment workflows"
    df = pd.DataFrame({"text_field1": [text1, text2], "text_field2": [text3, text4]})
    labeled_text_segments = extract_labeled_text_segments_from_keywords(
        df,
        ["text_field1", "text_field2"],
        {
            "test_label1": ["model", "track"],
            "test_label2": ["evaluation", "testing", "communicate"],
        },
    )
    actual_segments_row_1 = labeled_text_segments["labeled_text_segments"].iloc[0]
    actual_segments_row_2 = labeled_text_segments["labeled_text_segments"].iloc[1]
    expected_segments_row_1 = [
        LabeledTextSegment(
            text_field="text_field1",
            start=24,
            end=29,
            label="test_label1",
        ),
        LabeledTextSegment(
            text_field="text_field2",
            start=25,
            end=30,
            label="test_label1",
        ),
        LabeledTextSegment(
            text_field="text_field1",
            start=30,
            end=40,
            label="test_label2",
        ),
        LabeledTextSegment(
            text_field="text_field2",
            start=13,
            end=24,
            label="test_label2",
        ),
    ]

    expected_segments_row_2 = [
        LabeledTextSegment(
            text_field="text_field1",
            start=15,
            end=20,
            label="test_label1",
        ),
        LabeledTextSegment(
            text_field="text_field2",
            start=9,
            end=14,
            label="test_label1",
        ),
        LabeledTextSegment(
            text_field="text_field2",
            start=15,
            end=22,
            label="test_label2",
        ),
    ]

    assert actual_segments_row_1 == expected_segments_row_1
    assert actual_segments_row_2 == expected_segments_row_2
