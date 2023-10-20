import os
from argparse import ArgumentParser
from argparse import Namespace
from typing import Callable
from typing import Dict

import pandas as pd
from workflow import GroundTruth
from workflow import TestCase
from workflow import TestSample
from workflow import TestSuite
from tqdm import tqdm


import kolena
from kolena.workflow.annotation import ClassificationLabel

BUCKET = 'kolena-public-datasets'
DATASET = 'LibriSpeech'

def seed_test_suite_word_count(
    test_suite_name: str,
    complete_test_case: TestCase,
) -> TestSuite:
    test_case_name_to_decision_logic_map = {
        "short word count": lambda x: x < 11,
        "medium word count": lambda x: 11 <= x < 21,
        "long word count": lambda x: 21 <= x,
    }

    test_cases = []
    for name, fn in test_case_name_to_decision_logic_map.items():
        ts_list = [(ts, gt) for ts, gt in complete_test_case.iter_test_samples() if fn(ts.metadata['word_count'])]

        new_ts = TestCase(
            f"transcription word count :: {name} :: {DATASET}",
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

def seed_test_suite_duration(
    test_suite_name: str,
    complete_test_case: TestCase,
) -> TestSuite:
    test_case_name_to_decision_logic_map = {
        "short duration": lambda x: x < 4,
        "medium duration": lambda x: 4 <= x < 7,
        "long duration": lambda x: 7 <= x,
    }

    test_cases = []
    for name, fn in test_case_name_to_decision_logic_map.items():
        ts_list = [(ts, gt) for ts, gt in complete_test_case.iter_test_samples() if fn(ts.metadata['duration_seconds'])]

        new_ts = TestCase(
            f"transcription duration (seconds) :: {name} :: {DATASET}",
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

def seed_test_suite_speaker_sex(
    test_suite_name: str,
    complete_test_case: TestCase,
) -> TestSuite:
    test_case_name_to_decision_logic_map = {
        "male": lambda x: x == ' M ',
        "female": lambda x: x == ' F ',
    }

    test_cases = []
    for name, fn in test_case_name_to_decision_logic_map.items():
        ts_list = [(ts, gt) for ts, gt in complete_test_case.iter_test_samples() if fn(ts.metadata['speaker_sex'])]

        new_ts = TestCase(
            f"speaker sex:: {name} :: {DATASET}",
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

def seed_test_suite_longest_word_len(
    test_suite_name: str,
    complete_test_case: TestCase,
) -> TestSuite:
    test_case_name_to_decision_logic_map = {
        "short": lambda x: x < 7,
        "medium": lambda x: 7 <= x < 11,
        "long": lambda x: 11 <= x,
    }

    test_cases = []
    for name, fn in test_case_name_to_decision_logic_map.items():
        ts_list = [(ts, gt) for ts, gt in complete_test_case.iter_test_samples() if fn(ts.metadata['longest_word_len'])]

        new_ts = TestCase(
            f"longest word length :: {name} :: {DATASET}",
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

def seed_test_suite_max_pitch(
    test_suite_name: str,
    complete_test_case: TestCase,
) -> TestSuite:
    test_case_name_to_decision_logic_map = {
        "1st quartile": lambda x: x < 107.196,
        "2nd quartile": lambda x: 107.196 <= x < 107.206,
        "3rd quartile": lambda x: 107.206 <= x < 107.21,
        "4th quartile": lambda x: 107.21 <= x,
    }

    test_cases = []
    for name, fn in test_case_name_to_decision_logic_map.items():
        ts_list = [(ts, gt) for ts, gt in complete_test_case.iter_test_samples() if fn(ts.metadata['max_pitch'])]

        new_ts = TestCase(
            f"max pitch :: {name} :: {DATASET}",
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

def seed_test_suite_energy(
    test_suite_name: str,
    complete_test_case: TestCase,
) -> TestSuite:
    test_case_name_to_decision_logic_map = {
        "1st quartile": lambda x: x < 0.00207,
        "2nd quartile": lambda x: 0.00207 <= x < 0.00333,
        "3rd quartile": lambda x: 0.00333 <= x < 0.00485,
        "4th quartile": lambda x: 0.00485 <= x,
    }

    test_cases = []
    for name, fn in test_case_name_to_decision_logic_map.items():
        ts_list = [(ts, gt) for ts, gt in complete_test_case.iter_test_samples() if fn(ts.metadata['energy'])]

        new_ts = TestCase(
            f"energy :: {name} :: {DATASET}",
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

def seed_test_suite_zero_crossing_rate(
    test_suite_name: str,
    complete_test_case: TestCase,
) -> TestSuite:
    test_case_name_to_decision_logic_map = {
        "1st quartile": lambda x: x < 1771,
        "2nd quartile": lambda x: 1771 <= x < 2292,
        "3rd quartile": lambda x: 2292 <= x < 2838,
        "4th quartile": lambda x: 2838 <= x,
    }

    test_cases = []
    for name, fn in test_case_name_to_decision_logic_map.items():
        ts_list = [(ts, gt) for ts, gt in complete_test_case.iter_test_samples() if fn(ts.metadata['zero_crossing_rate'])]

        new_ts = TestCase(
            f"zero crossing rate :: {name} :: {DATASET}",
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

def seed_test_suite_tempo(
    test_suite_name: str,
    complete_test_case: TestCase,
) -> TestSuite:
    test_case_name_to_decision_logic_map = {
        "slower": lambda x: x < 2.5,
        "medium": lambda x: 2.5 <= x < 3.1,
        "faster": lambda x: 3.1 <= x,
    }

    test_cases = []
    for name, fn in test_case_name_to_decision_logic_map.items():
        ts_list = [(ts, gt) for ts, gt in complete_test_case.iter_test_samples() if fn(ts.metadata['tempo'])]

        new_ts = TestCase(
            f"tempo (words per second) :: {name} :: {DATASET}",
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

def seed_complete_test_case(args: Namespace) -> TestCase:
    df = pd.read_csv(args.dataset_csv)
    df = df.where(pd.notnull(df), None)  # read missing cells as None
    df.columns = df.columns.str.replace(r"(\s|\.)+", "_", regex=True)  # sanitize column names to use underscores

    required_columns = {'id', 'locator', 'text', 'inference_whisper_default',
                        'inference_wav2vec-base-960h', 'speaker_sex', 'duration_seconds',
                        'max_pitch', 'energy', 'zero_crossing_rate', 'word_count',
                        'longest_word_len', 'tempo'}
    assert all(required_column in set(df.columns) for required_column in required_columns)

    test_samples = []
    for record in tqdm(df.itertuples(index=False), total=len(df)):
        test_sample = TestSample(
            locator=f"s3://{BUCKET}/{DATASET}/{record.locator}",
            metadata={"file_id": record.id,
                      "text": record.text,
                      "speaker_sex": record.speaker_sex,
                      "duration_seconds": record.duration_seconds,
                      "max_pitch": record.max_pitch,
                      "energy": record.energy,
                      "zero_crossing_rate": record.zero_crossing_rate,
                      "word_count": record.word_count,
                      "longest_word_len": record.longest_word_len,
                      "tempo": record.tempo,
                      }
        )
        ground_truth = GroundTruth(transcription=ClassificationLabel(record.text))
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


def main(args: Namespace) -> None:
    kolena.initialize(os.environ['KOLENA_TOKEN'], verbose=True)
    complete_tc = seed_complete_test_case(args)

    test_suite_names: Dict[str, Callable[[str, TestCase], TestSuite]] = {
        # f"{DATASET} :: word count": seed_test_suite_word_count,
        # f"{DATASET} :: audio duration": seed_test_suite_duration,
        # f"{DATASET} :: speaker sex": seed_test_suite_speaker_sex,
        # f"{DATASET} :: longest word length": seed_test_suite_longest_word_len,
        # f"{DATASET} :: max pitch": seed_test_suite_max_pitch,
        # f"{DATASET} :: energy": seed_test_suite_energy,
        # f"{DATASET} :: zero crossing rate": seed_test_suite_zero_crossing_rate,
        f"{DATASET} :: tempo (words per second)": seed_test_suite_tempo,
    }
    seed_test_suites(test_suite_names, complete_tc)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset_csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/metadata.csv",
        help="CSV file specifying dataset. See default CSV for details",
    )

    main(ap.parse_args())