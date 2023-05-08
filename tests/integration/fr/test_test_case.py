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
from typing import List
from typing import Tuple

import pandas as pd
import pandas.testing
import pytest

from kolena.errors import NameConflictError
from kolena.errors import NotFoundError
from kolena.fr import TestCase
from kolena.fr import TestImages
from kolena.fr.datatypes import TestCaseRecord
from tests.integration.fr.conftest import TestData
from tests.integration.helper import with_test_prefix


@pytest.fixture(scope="module")
def with_augmented_images(fr_test_data: TestData) -> None:
    # re-register should not impact results
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


def assert_list_and_df_values_equal(lt: List[Tuple], df: pd.DataFrame) -> None:
    assert sorted(lt) == sorted(list(df.itertuples(index=False, name=None)))


def test__init() -> None:
    name = with_test_prefix(f"{__file__}::test__init test case")
    description = "some\ndescription\n\twith punctuation!"
    test_case = TestCase(name, description=description)
    assert test_case.name == name
    assert test_case.version == 0
    assert test_case.description == description

    test_case2 = TestCase(name)  # should re-load previously created
    assert test_case == test_case2

    test_case3 = TestCase(name, description="a different description")  # different description is ignored
    assert test_case == test_case3


def test__init__with_version(test_samples: List[TestCaseRecord]) -> None:
    name = with_test_prefix(f"{__file__}::test__init__with_version test case")
    test_case = TestCase(name, description="test")
    assert test_case.version == 0
    test_case0 = TestCase(name, version=test_case.version)  # reload with matching version
    assert test_case == test_case0

    with pytest.raises(NameConflictError):
        TestCase(name, version=123)  # invalid version throws

    with test_case.edit() as editor:
        editor.add(*test_samples[0])

    assert test_case.version == 1
    assert test_case == TestCase(name, version=test_case.version)
    assert test_case0 == TestCase(name, version=test_case0.version)


def test__init__with_test_samples(test_samples: List[TestCaseRecord]) -> None:
    name = with_test_prefix(f"{__file__}::test__init__with_test_samples test case")
    test_case = TestCase(name, test_samples=test_samples)
    assert test_case.version == 1
    assert_list_and_df_values_equal(test_samples, test_case.load_data())


def test__init__reset(test_samples: List[TestCaseRecord]) -> None:
    name = with_test_prefix(f"{__file__}::test__init__reset test case")
    description = f"{name} (description)"
    samples = [test_samples[3]]
    TestCase(name, description=description, test_samples=samples)

    new_samples = [test_samples[4]]
    test_case = TestCase(name, test_samples=new_samples, reset=True)
    assert test_case.version == 2
    assert test_case.description == description  # not updated or cleared
    assert_list_and_df_values_equal(new_samples, test_case.load_data())


def test__init__with_samples_reset(test_samples: List[TestCaseRecord]) -> None:
    name = with_test_prefix(f"{__file__}::test__init__with_samples_reset test case")
    samples = [test_samples[0], test_samples[3], test_samples[4]]
    test_case = TestCase(name, test_samples=samples, reset=True)
    assert test_case.version == 1
    assert_list_and_df_values_equal(samples, test_case.load_data())


def test_create() -> None:
    name = with_test_prefix(f"{__file__}::test_create test case")
    description = "A long\ndescription, with\tpunctuation $and all."
    test_case = TestCase.create(name, description=description)
    assert test_case.data.name == name
    assert test_case.data.version == 0
    assert test_case.data.description == description


def test_load() -> None:
    name = with_test_prefix(f"{__file__}::test_load test case")
    description = "123"
    TestCase.create(name, description=description)
    test_case = TestCase.load(name)
    assert test_case.data.name == name
    assert test_case.data.version == 0
    assert test_case.data.description == description
    assert test_case.data.image_count == 0
    assert test_case.data.pair_count_genuine == 0
    assert test_case.data.pair_count_imposter == 0


def test_load_does_not_exist() -> None:
    with pytest.raises(NotFoundError):
        TestCase.load("test_load_does_not_exist test case")


def test_load_data_empty() -> None:
    name = with_test_prefix(f"{__file__}::test_load_data_empty test case")
    test_case = TestCase.create(name)
    df = test_case.load_data()
    assert len(df) == 0


def test_iter_data_empty() -> None:
    name = with_test_prefix(f"{__file__}::test_iter_data_empty test case")
    test_case = TestCase.create(name)
    for _ in test_case.iter_data():
        pytest.fail("expected no data to iterate over")


def test_load_data(fr_test_data: TestData, fr_test_cases: List[TestCase]) -> None:
    test_case = fr_test_cases[0]
    image_pairs = fr_test_data.image_pairs
    expected = [
        image_pairs[0],
        image_pairs[1],
        image_pairs[8],
        image_pairs[9],
    ]
    assert_list_and_df_values_equal(expected, test_case.load_data())


def test_iter_data(fr_test_data: TestData, fr_test_cases: List[TestCase]) -> None:
    test_case = fr_test_cases[0]
    image_pairs = fr_test_data.image_pairs
    expected = [
        image_pairs[0],
        image_pairs[1],
        image_pairs[8],
        image_pairs[9],
    ]
    frames = []
    for df in test_case.iter_data(batch_size=2):
        frames.append(df)
    df_test_case_loaded = pd.concat(frames)
    assert_list_and_df_values_equal(expected, df_test_case_loaded)


def test_edit(fr_test_data: TestData, with_augmented_images: None) -> None:
    name = with_test_prefix(f"{__file__}::test_edit test case")
    test_case = TestCase.create(name)
    assert test_case.data.version == 0

    new_description = "some new description"
    augmented_images = fr_test_data.augmented_images
    with test_case.edit() as editor:
        editor.description(new_description)

        original_locators = [aug[4] for aug in augmented_images]
        augmented_locators = [aug[0] for aug in augmented_images]
        editor.add(original_locators[0], original_locators[1], True)
        editor.add(original_locators[2], augmented_locators[3], False)
        editor.add(original_locators[3], original_locators[4], False)
        editor.add(original_locators[2], original_locators[5], False)

        editor.remove(original_locators[3], original_locators[4])

    assert test_case.data.version == 1
    assert test_case.data.description == new_description
    assert test_case.data.image_count == 5
    assert test_case.data.pair_count_genuine == 1
    assert test_case.data.pair_count_imposter == 2

    sort_column = "locator_b"
    df = test_case.load_data().sort_values(by=sort_column, ignore_index=True)
    df_expected = pd.DataFrame(
        dict(
            locator_a=[original_locators[0], original_locators[2], original_locators[2]],
            locator_b=[original_locators[1], augmented_locators[3], original_locators[5]],
            is_same=[True, False, False],
        ),
    ).sort_values(by=sort_column, ignore_index=True)
    pandas.testing.assert_frame_equal(df, df_expected)

    with pytest.raises(KeyError):
        with test_case.edit() as editor:
            editor.remove("foo", "bar")

    # loading by name loads the latest version
    test_case = TestCase.load(name)
    assert test_case.data.version == 1
    assert len(test_case.load_data()) == 3

    # loading by explicit version works
    test_case = TestCase.load(name, version=0)
    assert test_case.data.version == 0
    assert len(test_case.load_data()) == 0


def test_edit_no_op() -> None:
    name = with_test_prefix(f"{__file__}::test_edit_no_op test case")
    description = f"{name} (description)"
    test_case = TestCase.create(name, description)
    version = 0

    with test_case.edit():
        ...
    assert test_case.data.version == version

    with test_case.edit() as editor:
        editor.description("a new description")
        editor.description(description)
    assert test_case.data.version == version

    sample = "s3://bucket/img1.png", "s3://bucket/img2.png", True
    with test_case.edit() as editor:
        editor.add(*sample)
        editor.remove(*sample[:2])
    assert test_case.data.version == version
    assert test_case.data.image_count == 0
    assert test_case.data.pair_count_genuine == 0
    assert test_case.data.pair_count_imposter == 0


def test_edit_empty(fr_images_df: pd.DataFrame) -> None:
    name = with_test_prefix(f"{__file__}::test_edit_empty test case")
    test_case = TestCase.create(name)
    sample = fr_images_df.iloc[0]["locator"], fr_images_df.iloc[1]["locator"], True

    with test_case.edit() as editor:
        editor.description("description")
    assert test_case.data.version == 1
    assert test_case.data.description == "description"

    # add a sample to the test case for later removal
    with test_case.edit() as editor:
        editor.add(*sample)
    assert test_case.data.version == 2
    assert test_case.data.image_count == 2
    assert test_case.data.pair_count_genuine == 1
    assert test_case.data.pair_count_imposter == 0

    # empty the test case
    with test_case.edit() as editor:
        editor.remove(*sample[:2])
    assert test_case.data.version == 3
    assert test_case.data.image_count == 0
    assert test_case.data.pair_count_genuine == 0
    assert test_case.data.pair_count_imposter == 0


def test__edit__add(test_samples: List[TestCaseRecord]) -> None:
    name = with_test_prefix(f"{__file__}::test__edit__add test case")
    description = f"{name} (description)"
    samples_0 = [test_samples[0], test_samples[3], test_samples[4], test_samples[5]]
    samples_1 = [test_samples[1], test_samples[2], test_samples[3]]
    test_case = TestCase(name, description=description, test_samples=samples_0)

    with test_case.edit() as editor:
        for test_sample in samples_1:
            editor.add(*test_sample)
    assert test_case.version == 2
    assert test_case.description == description
    assert_list_and_df_values_equal(test_samples[:6], test_case.load_data())
    assert test_case.image_count == 3
    assert test_case.pair_count_genuine == 2
    assert test_case.pair_count_imposter == 4


def test__edit__remove(test_samples: List[TestCaseRecord]) -> None:
    name = with_test_prefix(f"{__file__}::test__edit__remove test case")
    description = f"{name} (description)"
    samples = [test_samples[0], test_samples[3], test_samples[4], test_samples[5]]
    test_case = TestCase(name, description=description, test_samples=samples)

    with pytest.raises(KeyError):
        with test_case.edit() as editor:
            editor.remove(*test_samples[1][:2])

    with test_case.edit() as editor:
        for i in range(2):
            editor.remove(*samples[i][:2])
    assert test_case.version == 2
    assert test_case.description == description
    assert_list_and_df_values_equal(samples[2:], test_case.load_data())
    assert test_case.image_count == 2
    assert test_case.pair_count_genuine == 1
    assert test_case.pair_count_imposter == 1


def test__edit__reset(test_samples: List[TestCaseRecord]) -> None:
    name = with_test_prefix(f"{__file__}::test__edit__reset test case")
    description = f"{name} (description)"
    samples_0 = [test_samples[0], test_samples[3], test_samples[4], test_samples[5]]
    samples_1 = [test_samples[1], test_samples[2], test_samples[3]]
    test_case = TestCase(name, description=description, test_samples=samples_0)

    # the reset flag will bump the version even the description doesn't change
    with test_case.edit(reset=True) as editor:
        editor.description(description)
    assert test_case.version == 2
    assert test_case.description == description
    assert_list_and_df_values_equal([], test_case.load_data())

    with test_case.edit(reset=True) as editor:
        for test_sample in samples_1:
            editor.add(*test_sample)

    assert test_case.version == 3
    assert test_case.description == description
    assert_list_and_df_values_equal(samples_1, test_case.load_data())
    assert test_case.image_count == 3
    assert test_case.pair_count_genuine == 0
    assert test_case.pair_count_imposter == 3
