import os
from argparse import Namespace
from collections.abc import Iterator

import pytest
from text_summarization.seed_test_run import run as seed_test_run_main
from text_summarization.seed_test_suite import run as seed_test_suite_main

from kolena._utils.state import kolena_session


@pytest.fixture(scope="session", autouse=True)
def with_init() -> Iterator[None]:
    with kolena_session(api_token=os.environ["KOLENA_TOKEN"]):
        yield


def test__seed_test_suite() -> None:
    args = Namespace(dataset_csv="s3://kolena-public-datasets/CNN_DailyNews/metadata/CNN_DailyMail_metadata_100.csv")
    seed_test_suite_main(args)


@pytest.mark.depends(on=["test__seed_test_suite"])
def test__seed_test_run() -> None:
    args = Namespace(model_name="ada", test_suite="CNN-DailyMail :: text length", local_csv="none")
    seed_test_run_main(args)
