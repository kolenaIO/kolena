from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import pytest

from kolena.detection import TestCase as DetectionTestCase
from kolena.errors import WorkflowMismatchError
from tests.integration.generic.dummy import DUMMY_WORKFLOW
from tests.integration.generic.dummy import DummyGroundTruth
from tests.integration.generic.dummy import DummyTestSample
from tests.integration.generic.dummy import TestCase
from tests.integration.helper import assert_sorted_list_equal
from tests.integration.helper import with_test_prefix


def assert_test_case(test_case: TestCase, name: str, version: int) -> None:
    assert test_case.workflow == DUMMY_WORKFLOW
    assert test_case.name == name
    assert test_case.version == version


def test__init(
    with_init: None,
    dummy_test_samples: List[DummyTestSample],
    dummy_ground_truths: List[DummyGroundTruth],
) -> None:
    name = with_test_prefix("Generic::test__init test case")
    assert_test_case(TestCase(name), name, 0)  # should create
    assert_test_case(TestCase(name), name, 0)  # should load

    all_test_samples = list(zip(dummy_test_samples, dummy_ground_truths))
    test_case = TestCase(name, test_samples=all_test_samples)
    assert test_case.version == 1
    assert_sorted_list_equal(test_case.load_test_samples(), all_test_samples)


def test__init__reset(
    with_init: None,
    dummy_test_samples: List[DummyTestSample],
    dummy_ground_truths: List[DummyGroundTruth],
) -> None:
    name = with_test_prefix("Generic::test__init__reset test case")
    description = f"{name} (description)"
    TestCase(name, description=description, test_samples=list(zip(dummy_test_samples, dummy_ground_truths)))

    new_test_samples = list(zip(dummy_test_samples[:2][::-1], dummy_ground_truths[:2][::1]))
    test_case = TestCase(name, test_samples=new_test_samples, reset=True)
    assert test_case.version == 2
    assert test_case.description == description  # not updated or cleared
    assert sorted(test_case.load_test_samples()) == sorted(new_test_samples)


# @pytest.mark.internal
@pytest.mark.skip
def test__init__reset_with_overlap(
    with_init: None,
    dummy_test_samples: List[DummyTestSample],
    dummy_ground_truths: List[DummyGroundTruth],
) -> None:
    name = with_test_prefix("Generic::test__init__reset_with_overlap test case")
    description = f"{name} (description)"
    sample_batch_1 = list(zip(dummy_test_samples[:6], dummy_ground_truths[:6]))
    sample_batch_2 = list(zip(dummy_test_samples[4:], dummy_ground_truths[4:]))
    TestCase(name, description=description, test_samples=sample_batch_1)

    test_case = TestCase(name, test_samples=sample_batch_2, reset=True)
    assert test_case.version == 2
    assert test_case.description == description  # not updated or cleared
    # overlapping samples / gts should be preserved
    assert sorted(test_case.load_test_samples()) == sorted(sample_batch_2)


@pytest.mark.skip
def test__init__reset_with_other_test_case(
    with_init: None,
    dummy_test_samples: List[DummyTestSample],
    dummy_ground_truths: List[DummyGroundTruth],
) -> None:
    name = with_test_prefix("Generic::test__init__reset_with_other_test_case test case")
    name_other = with_test_prefix("Generic::test__init__reset_with_other_test_case test case (other)")
    description = f"{name} (description)"
    sample_batch_1 = list(zip(dummy_test_samples, dummy_ground_truths))
    sample_batch_2 = list(zip(dummy_test_samples[:4], dummy_ground_truths[:4]))
    sample_batch_3 = list(zip(dummy_test_samples[:2][::-1], dummy_ground_truths[:2][::1]))

    # Create and update test case
    TestCase(name, description=description, test_samples=sample_batch_1)
    test_case_other = TestCase(name_other, test_samples=sample_batch_2)

    test_case = TestCase(name, test_samples=sample_batch_3, reset=True)
    assert test_case.version == 2
    assert test_case.description == description  # not updated or cleared
    # sample_batch_2 should be untouched
    assert_sorted_list_equal(test_case_other.load_test_samples(), sample_batch_2)
    assert_sorted_list_equal(test_case.load_test_samples(), sample_batch_3)  # sample_batch_1 should be cleared


@pytest.mark.skip
def test__init__reset_resets_all_past_samples(
    with_init: None,
    dummy_test_samples: List[DummyTestSample],
    dummy_ground_truths: List[DummyGroundTruth],
) -> None:
    name = with_test_prefix("Generic::test__init__reset_resets_all_past_samples test case")
    description = f"{name} (description)"
    sample_batch_1 = list(zip(dummy_test_samples[:2], dummy_ground_truths[:2]))
    sample_batch_2 = list(zip(dummy_test_samples[2:4], dummy_ground_truths[2:4]))
    sample_batch_3 = list(zip(dummy_test_samples[4:], dummy_ground_truths[4:]))

    # Create and update test case
    initial_test_case = TestCase(name, description=description, test_samples=sample_batch_1)
    with initial_test_case.edit() as editor:
        for sample, gt in sample_batch_2:
            editor.add(sample, gt)

    test_case = TestCase(name, test_samples=sample_batch_3, reset=True)
    assert test_case.version == 3
    assert test_case.description == description  # not updated or cleared
    # both sample_batch_1 and sample_batch_2 should be cleared
    assert_sorted_list_equal(test_case.load_test_samples(), sample_batch_3)


def test__create(with_init: None) -> None:
    name = with_test_prefix("Generic::test__create test case")
    assert_test_case(TestCase.create(name), name, 0)

    with pytest.raises(Exception):  # TODO(gh): better error?
        TestCase.create(name)


def test__load(with_init: None) -> None:
    name = with_test_prefix("Generic::test__load test case")

    with pytest.raises(Exception):  # TODO(gh): better error?
        TestCase.load(name)

    TestCase.create(name)
    assert_test_case(TestCase.load(name), name, 0)


def test__load__mismatching_workflows(with_init: None) -> None:
    name = with_test_prefix("Generic::test__load__mismatching_workflows")
    DetectionTestCase(name)
    with pytest.raises(WorkflowMismatchError):
        TestCase(name)


def test__edit(
    with_init: None,
    dummy_test_samples: List[DummyTestSample],
    dummy_ground_truths: List[DummyGroundTruth],
) -> None:
    test_case = TestCase(with_test_prefix("Generic::test__edit test case"))
    assert test_case.version == 0

    description = "new description"
    all_samples = list(zip(dummy_test_samples, dummy_ground_truths))
    with test_case.edit() as editor:
        editor.description(description)
        for test_sample, ground_truth in all_samples:
            editor.add(test_sample, ground_truth)

    assert test_case.version == 1
    assert test_case.description == description
    assert sorted(test_case.load_test_samples()) == sorted(all_samples)


def test__edit__reset(
    with_init: None,
    dummy_test_samples: List[DummyTestSample],
    dummy_ground_truths: List[DummyGroundTruth],
) -> None:
    test_case = TestCase(
        with_test_prefix("Generic::test__edit__reset test case"),
        test_samples=[(dummy_test_samples[0], dummy_ground_truths[0]), (dummy_test_samples[1], dummy_ground_truths[1])],
    )

    added = [
        (dummy_test_samples[2], dummy_ground_truths[2]),
        (dummy_test_samples[1], dummy_ground_truths[3]),  # re-add sample that was previously present
    ]
    with test_case.edit(reset=True) as editor:
        for tc, gt in added:
            editor.add(tc, gt)

    assert test_case.version == 2
    assert sorted(test_case.load_test_samples()) == sorted(added)


def test__edit__replace(
    with_init: None,
    dummy_test_samples: List[DummyTestSample],
    dummy_ground_truths: List[DummyGroundTruth],
) -> None:
    test_case = TestCase(with_test_prefix("Generic::test__edit__replace test case"))

    # one pass, first is shadowed
    with test_case.edit() as editor:
        editor.add(dummy_test_samples[0], dummy_ground_truths[0])
        editor.add(dummy_test_samples[0], dummy_ground_truths[1])

    assert test_case.load_test_samples() == [(dummy_test_samples[0], dummy_ground_truths[1])]

    # two passes
    with test_case.edit() as editor:
        editor.add(dummy_test_samples[0], dummy_ground_truths[2])

    assert sorted(test_case.load_test_samples()) == sorted([(dummy_test_samples[0], dummy_ground_truths[2])])


def test__edit__remove(
    with_init: None,
    dummy_test_samples: List[DummyTestSample],
    dummy_ground_truths: List[DummyGroundTruth],
) -> None:
    test_case = TestCase(
        with_test_prefix("Generic::test__edit__remove test case"),
        test_samples=[
            (dummy_test_samples[0], dummy_ground_truths[0]),
            (dummy_test_samples[1], dummy_ground_truths[1]),
        ],
    )

    with test_case.edit() as editor:
        editor.remove(dummy_test_samples[0])
        editor.add(dummy_test_samples[2], dummy_ground_truths[0])

    assert test_case.version == 2
    assert test_case.load_test_samples() == [
        (dummy_test_samples[1], dummy_ground_truths[1]),
        (dummy_test_samples[2], dummy_ground_truths[0]),
    ]

    with test_case.edit() as editor:
        editor.remove(dummy_test_samples[0])  # removing sample not in case is fine
        editor.remove(dummy_test_samples[1])
        editor.remove(dummy_test_samples[1])  # removing sample twice is fine
        editor.add(dummy_test_samples[1], dummy_ground_truths[2])  # add sample back in

    assert test_case.version == 3
    assert test_case.load_test_samples() == [
        (dummy_test_samples[2], dummy_ground_truths[0]),
        (dummy_test_samples[1], dummy_ground_truths[2]),
    ]


def test__edit__remove_only(
    with_init: None,
    dummy_test_samples: List[DummyTestSample],
    dummy_ground_truths: List[DummyGroundTruth],
) -> None:
    ts0, *ts_rest = dummy_test_samples
    gt0, *gt_rest = dummy_ground_truths
    rest = list(zip(ts_rest, gt_rest))
    test_case = TestCase(
        with_test_prefix("Generic::test__edit__remove_only test case"),
        test_samples=[(ts0, gt0), *rest],
    )

    with test_case.edit() as editor:
        editor.remove(ts0)

    assert test_case.version == 2
    assert sorted(test_case.load_test_samples()) == sorted(rest)


@pytest.mark.skip
def test__edit__description_only(
    with_init: None,
    dummy_test_samples: List[DummyTestSample],
    dummy_ground_truths: List[DummyGroundTruth],
) -> None:
    all_test_samples = list(zip(dummy_test_samples, dummy_ground_truths))
    test_case = TestCase(
        with_test_prefix("Generic::test__edit__description_only test case"),
        test_samples=all_test_samples,
    )

    description = "new description"
    with test_case.edit() as editor:
        editor.description(description)

    assert test_case.version == 2
    assert test_case.description == description
    assert_sorted_list_equal(test_case.load_test_samples(), all_test_samples)


@pytest.mark.skip
def test__edit__update_sample_metadata(
    with_init: None,
    dummy_test_samples: List[DummyTestSample],
    dummy_ground_truths: List[DummyGroundTruth],
) -> None:
    test_case_name = with_test_prefix("Generic::test__edit__update_sample_metadata test case")
    all_test_samples_v1 = list(zip(dummy_test_samples, dummy_ground_truths))
    test_case = TestCase(test_case_name, test_samples=all_test_samples_v1)
    original_version = test_case.version
    assert_sorted_list_equal(test_case.load_test_samples(), all_test_samples_v1)

    def generate_samples_with_ground_truths(
        generate_metadata: Callable[[int], Dict[str, int]],
    ) -> List[Tuple[DummyTestSample, DummyGroundTruth]]:
        test_samples = [
            DummyTestSample(  # type: ignore
                locator=sample.locator,
                bbox=sample.bbox,
                value=sample.value,
                metadata=generate_metadata(i),
            )
            for i, sample in enumerate(dummy_test_samples)
        ]
        return list(zip(test_samples, dummy_ground_truths))

    all_test_samples_v2 = generate_samples_with_ground_truths(
        lambda i: {"index": i, "second_index": i},
    )
    with test_case.edit() as editor:
        for sample, gt in all_test_samples_v2:
            editor.add(sample, gt)
    assert_sorted_list_equal(test_case.load_test_samples(), all_test_samples_v2)

    # validate that adding new metadata fields merges the metadata
    all_test_samples_v3 = generate_samples_with_ground_truths(
        lambda i: {"second_index": i * 2, "third_index": i * 3},
    )
    all_test_samples_merged_metadata = generate_samples_with_ground_truths(
        lambda i: {"index": i, "second_index": i * 2, "third_index": i * 3},
    )
    with test_case.edit() as editor:
        for sample, gt in all_test_samples_v3:
            editor.add(sample, gt)
    assert_sorted_list_equal(test_case.load_test_samples(), all_test_samples_merged_metadata)

    # test samples from original version now also have updated metadata
    original_test_case = TestCase(test_case_name, version=original_version)
    assert_sorted_list_equal(original_test_case.load_test_samples(), all_test_samples_merged_metadata)
