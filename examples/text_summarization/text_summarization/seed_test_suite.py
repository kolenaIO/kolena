import os
from argparse import ArgumentParser
from argparse import Namespace
from typing import Callable
from typing import Dict

import pandas as pd
from tqdm import tqdm

import kolena
from .utils import get_readable
from .workflow import GroundTruth
from .workflow import TestCase
from .workflow import TestSample
from .workflow import TestSuite

DATASET = "CNN-DailyMail"


def seed_test_suite_by_text(
    test_suite_name: str,
    complete_test_case: TestCase,
) -> TestSuite:
    test_case_name_to_decision_logic_map = {
        "short": lambda x: x < 500,
        "medium": lambda x: 500 <= x < 1000,
        "long": lambda x: 1000 <= x,
    }

    test_cases = []
    for name, fn in test_case_name_to_decision_logic_map.items():
        ts_list = [(ts, gt) for ts, gt in complete_test_case.iter_test_samples() if fn(ts.word_count)]

        new_ts = TestCase(
            f"text length :: {name} :: {DATASET}",
            test_samples=ts_list,
            reset=True,
        )
        test_cases.append(new_ts)

    test_suite = TestSuite(
        test_suite_name,
        test_cases=[complete_test_case, *test_cases],
        reset=True,
    )
    print(f"created test suite {test_suite.name} v{test_suite.version}")


def seed_test_suite_by_text_x_gt(
    test_suite_name: str,
    complete_test_case: TestCase,
) -> TestSuite:
    test_case_name_to_decision_logic_map = {
        "short text + short GT": lambda x, y: x < 500 and y < 60,
        "short text + long GT": lambda x, y: x < 500 and y >= 60,
        "medium text + short GT": lambda x, y: 500 <= x < 1000 and y < 60,
        "medium text + long GT": lambda x, y: 500 <= x < 1000 and y >= 60,
        "long text + short GT": lambda x, y: 1000 <= x and y < 60,
        "long text + long GT": lambda x, y: 1000 <= x and y >= 60,
    }

    test_cases = []
    for name, fn in test_case_name_to_decision_logic_map.items():
        ts_list = [(ts, gt) for ts, gt in complete_test_case.iter_test_samples() if fn(ts.word_count, gt.word_count)]

        new_ts = TestCase(
            f"text X GT length :: {name} :: {DATASET}",
            test_samples=ts_list,
            reset=True,
        )
        test_cases.append(new_ts)

    test_suite = TestSuite(
        test_suite_name,
        test_cases=[complete_test_case, *test_cases],
        reset=True,
    )
    print(f"created test suite {test_suite.name} v{test_suite.version}")


def seed_test_suite_by_category(
    test_suite_name: str,
    complete_test_case: TestCase,
) -> TestSuite:
    complete_test_samples = complete_test_case.load_test_samples()
    categories = list({ts.metadata["category"] for ts, _ in complete_test_samples})
    test_cases = []
    for category in categories:
        samples = [(ts, gt) for ts, gt in complete_test_samples if ts.metadata["category"] == category]

        ts = TestCase(
            f"news category :: {category} :: {DATASET}",
            test_samples=samples,
            reset=True,
        )
        test_cases.append(ts)

    test_suite = TestSuite(
        test_suite_name,
        test_cases=[complete_test_case, *test_cases],
        reset=True,
    )
    print(f"created test suite {test_suite.name} v{test_suite.version}")


def seed_test_suite_by_moderation(
    test_suite_name: str,
    complete_test_case: TestCase,
) -> TestSuite:
    test_case_name_to_decision_logic_map = {
        "low": lambda x: x < 0.01,
        "medium": lambda x: 0.01 <= x < 0.04,
        "high": lambda x: 0.04 <= x < 0.22,
        "very high": lambda x: 0.22 <= x,
    }

    test_cases = []
    for name, fn in test_case_name_to_decision_logic_map.items():
        samples = [(ts, gt) for ts, gt in complete_test_case.iter_test_samples() if fn(ts.metadata["moderation_score"])]

        ts = TestCase(
            f"moderation score :: {name} :: {DATASET}",
            test_samples=samples,
            reset=True,
        )
        test_cases.append(ts)

    test_suite = TestSuite(
        test_suite_name,
        test_cases=[complete_test_case, *test_cases],
        reset=True,
    )
    print(f"created test suite {test_suite.name} v{test_suite.version}")


def seed_complete_test_case(args: Namespace) -> TestCase:
    df = pd.read_csv(args.dataset_csv)
    df = df.where(pd.notnull(df), None)  # read missing cells as None
    df.columns = df.columns.str.replace(r"(\s|\.)+", "_", regex=True)  # sanitize column names to use underscores
    required_columns = {"article_id", "article", "article_summary", "text_word_count", "summary_word_count"}
    assert all(required_column in set(df.columns) for required_column in required_columns)
    optional_columns = {
        "prediction",
        "prediction_time",
        "tokens_text",
        "tokens_summary",
        "tokens_generated",
        "tokens_used",
        "cost",
    }

    test_samples = []
    for record in tqdm(df.itertuples(index=False), total=len(df)):
        test_sample = TestSample(  # type: ignore
            text=record.article,
            id=record.article_id,
            word_count=record.text_word_count,
            metadata={f: getattr(record, f) for f in set(record._fields) - required_columns - optional_columns},
        )
        ground_truth = GroundTruth(summary=get_readable(record.article_summary), word_count=record.summary_word_count)
        test_samples.append((test_sample, ground_truth))

    test_case = TestCase(f"complete :: {DATASET}", test_samples=test_samples, reset=True)
    print(f"Created test case: {test_case}")

    return test_case


def seed_test_suites(
    test_suite_names: Dict[str, Callable[[str, TestCase], TestSuite]],
    complete_test_case: TestCase,
) -> None:
    test_suites = []

    for test_suite_name, test_suite_fn in test_suite_names.items():
        test_suites.append(test_suite_fn(test_suite_name, complete_test_case))

    return None


def run(args: Namespace) -> None:
    complete_tc = seed_complete_test_case(args)

    test_suite_names: Dict[str, Callable[[str, TestCase], TestSuite]] = {
        f"{DATASET} :: text length": seed_test_suite_by_text,
        f"{DATASET} :: moderation score": seed_test_suite_by_moderation,
        f"{DATASET} :: text X ground truth length": seed_test_suite_by_text_x_gt,
        f"{DATASET} :: news category": seed_test_suite_by_category,
    }
    seed_test_suites(test_suite_names, complete_tc)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset_csv",
        type=str,
        default="s3://kolena-public-datasets/CNN-DailyMail/metadata/CNN_DailyMail_metadata.csv",
        help="CSV file specifying dataset. See default CSV for details",
    )

    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
