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
import re

import pandas as pd

from kolena.annotation import LabeledTextSegment


def _find_substring_indices(field_name: str, text: str, substrings: list[str], label: str) -> list[LabeledTextSegment]:
    segments = []
    for substring in substrings:
        escaped_search = re.escape(substring)

        pattern = rf"\b{escaped_search}\b"

        matches = re.finditer(pattern, text, re.IGNORECASE)

        segments.extend(
            [
                LabeledTextSegment(text_field=field_name, start=match.start(), end=match.end(), label=label)
                for match in matches
            ],
        )

    return segments


def _extract_labeled_text_segments_from_keywords_single_row(
    texts: dict[str, str],
    keyword_labels: dict[str, list[str]],
) -> list[LabeledTextSegment]:
    labeled_segments = []
    for label, keywords in keyword_labels.items():
        for field_name, text in texts.items():
            labeled_segments.extend(_find_substring_indices(field_name, text, keywords, label))
    return labeled_segments


def extract_labeled_text_segments_from_keywords(
    df: pd.DataFrame,
    text_fields: list[str],
    keyword_labels: dict[str, list[str]],
    labeled_text_segments_column: str = "labeled_text_segments",
) -> pd.DataFrame:
    """
    Extracts and labels text segments from specified text fields within a DataFrame based on given keywords.
    The resulting labeled segments are stored in a new column.

    :param df: DataFrame containing the text data.
    :param text_fields: Names of the columns in `df` that contain the text to be analyzed.
    :param keyword_labels: A dictionary where each key is a label and each value is a list of keywords associated with
     that label. Any text segment containing a keyword will be tagged with the corresponding label.
    :param labeled_text_segments_column: The name of the column in `df` where the resulting labeled text segments
        will be stored. Defaults to "labeled_text_segments".

    :return: The modified DataFrame with an additional column containing the labeled text segments.

    """
    df[labeled_text_segments_column] = df[text_fields].apply(
        lambda texts: _extract_labeled_text_segments_from_keywords_single_row(
            texts.to_dict(),
            keyword_labels,
        ),
        axis=1,
    )
    return df
