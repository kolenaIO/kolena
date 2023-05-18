import os
from argparse import Namespace
from collections.abc import Iterator

import pytest
from keypoint_detection.seed_test_run import run as seed_test_run_main
from keypoint_detection.seed_test_suite import run as seed_test_suite_main

from kolena._utils.state import kolena_session


@pytest.fixture(scope="session", autouse=True)
def with_init() -> Iterator[None]:
    with kolena_session(api_token=os.environ["KOLENA_TOKEN"]):
        yield


def test__seed_test_suite() -> None:
    args = Namespace(test_suite="300-W :: complete")
    seed_test_suite_main(args)


@pytest.mark.depends(on=["test__seed_test_suite"])
def test__seed_test_run() -> None:
    args = Namespace(model_name="Point Randomizer", test_suite="300-W :: complete")
    seed_test_run_main(args)
