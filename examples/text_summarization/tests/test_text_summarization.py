from argparse import Namespace

import pytest
from text_summarization.seed_test_run import seed_test as seed_test_run_main
from text_summarization.seed_test_suite import main as seed_test_suite_main


def test__seed_test_suite() -> None:
    seed_test_suite_main()


@pytest.mark.depends(on=["test__seed_test_suite"])
def test__seed_test_run() -> None:
    seed_test_run_main(Namespace(model_name="ada"))
