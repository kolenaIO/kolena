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
from typing import List

import numpy as np
import pandas as pd
import pandas.testing
import pytest

from kolena.errors import InputValidationError
from kolena.fr import TestCase
from kolena.fr import TestImages
from kolena.fr import TestSuite
from tests.integration.fr.conftest import TestData
from tests.integration.helper import with_test_prefix


def test__register(fr_test_data: TestData) -> None:
    # normal registration covered in fr_test_data fixture

    with pytest.raises(InputValidationError):
        with TestImages.register() as registrar:
            registrar.add("garbage locator", "anything", 10, 10)

    # dimensions must be positive
    with pytest.raises(InputValidationError):
        with TestImages.register() as registrar:
            registrar.add("s3://bucket/image.png", "anything", 0, 0)


def test__register__no_op() -> None:
    with TestImages.register():  # should not throw or do anything
        ...


def test__iter(fr_test_data: TestData, fr_images_df: pd.DataFrame) -> None:
    data_source = fr_test_data.data_sources[0]
    for df_test_image in TestImages.iter(data_source=data_source, batch_size=2):
        assert len(df_test_image) <= 2

    df_test_image_original_source = fr_images_df[fr_images_df["data_source"] == data_source]
    for df_test_image in TestImages.iter(
        data_source=data_source,
        batch_size=len(df_test_image_original_source),
    ):
        pd.testing.assert_frame_equal(df_test_image.drop(columns="image_id"), df_test_image_original_source)


def test__load(fr_test_data: TestData, fr_images_df: pd.DataFrame) -> None:
    data_source = fr_test_data.data_sources[0]
    df_test_image = TestImages.load(data_source=data_source).drop(columns="image_id")
    df_test_image_original_source = fr_images_df[fr_images_df["data_source"] == data_source]
    pandas.testing.assert_frame_equal(df_test_image, df_test_image_original_source)

    df_test_image = TestImages.load(data_source="data source that does not exist")
    assert len(df_test_image) == 0


def test__load__test_suite(fr_test_suites: List[TestSuite]) -> None:
    test_suite = TestSuite.load_by_name(with_test_prefix("A"))
    df_test_image = TestImages.load(test_suite)
    df_test_image2 = TestImages.load(test_suite.data)
    pandas.testing.assert_frame_equal(df_test_image, df_test_image2)


def test__load__test_case(fr_test_suites: List[TestSuite]) -> None:
    test_case = TestCase.load_by_name(with_test_prefix("A"))
    df_test_image = TestImages.load(test_case)
    df_test_image2 = TestImages.load(test_case.data)
    pandas.testing.assert_frame_equal(df_test_image, df_test_image2)


def test__register__augmented(fr_test_data: TestData) -> None:
    with TestImages.register() as registrar:
        for record in fr_test_data.augmented_images:
            registrar.add_augmented(
                record[4],
                record[0],
                record[5],
                width=record[2],
                height=record[3],
                bounding_box=record[6],
                landmarks=record[7],
                tags=record[8],
            )


@pytest.mark.depends(on=["test__register__augmented"])
def test__load__augmented(
    fr_test_data: TestData,
    fr_images_df: pd.DataFrame,
    fr_augmented_images_df: pd.DataFrame,
    fr_augmented_images_expected_df: pd.DataFrame,
) -> None:
    data_source_1, data_source_2 = fr_test_data.data_sources

    df_test_image_1 = TestImages.load(include_augmented=True, data_source=data_source_1).drop(columns="image_id")
    df_test_image_2 = TestImages.load(include_augmented=True, data_source=data_source_2).drop(columns="image_id")

    df_test_image_augmented_1 = fr_augmented_images_expected_df[
        fr_augmented_images_expected_df["data_source"] == data_source_1
    ]
    pandas.testing.assert_frame_equal(df_test_image_1.dropna(subset=["original_locator"]), df_test_image_augmented_1)

    df_test_image = pd.concat([df_test_image_1, df_test_image_2], ignore_index=True)
    assert len(df_test_image) == len(fr_images_df) + len(fr_augmented_images_df)
    df_test_image_augmented = df_test_image.dropna(subset=["original_locator"]).reset_index(drop=True)
    pandas.testing.assert_frame_equal(df_test_image_augmented, fr_augmented_images_expected_df)

    df_test_image = TestImages.load(include_augmented=True, data_source="data source that does not exist")
    assert len(df_test_image) == 0


@pytest.mark.depends(on=["test__load__augmented"])
def test__register__update(fr_images_df: pd.DataFrame) -> None:
    with TestImages.register() as registrar:
        record = fr_images_df.iloc[0]
        registrar.add(
            record.locator,
            record.data_source,
            record.width + 10,
            record.height + 20,
            bounding_box=record.bounding_box + 10,
            landmarks=None,  # previous value should get propagated
            tags=dict(new_category="new_value"),
        )

    df_test_image = TestImages.load()
    df_updated = df_test_image[df_test_image["locator"] == record.locator]
    assert len(df_updated) == 1
    record_updated = df_updated.iloc[0]
    assert record_updated.width == record.width + 10
    assert record_updated.height == record.height + 20
    assert np.allclose(record_updated.bounding_box, record.bounding_box + 10)
    assert record_updated.landmarks == record.landmarks
    assert record_updated.tags == {**record.tags, "new_category": "new_value"}
