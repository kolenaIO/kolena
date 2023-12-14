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
from argparse import ArgumentParser
from argparse import Namespace
from typing import Callable
from typing import Dict

import pandas as pd
from automatic_speech_recognition.workflow import GroundTruth
from automatic_speech_recognition.workflow import TestCase
from automatic_speech_recognition.workflow import TestSample
from automatic_speech_recognition.workflow import TestSuite
from tqdm import tqdm

import kolena
from kolena.workflow.annotation import ClassificationLabel

BUCKET = "kolena-public-datasets"
DATASET = "LibriSpeech"


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
        ts_list = [(ts, gt) for ts, gt in complete_test_case.iter_test_samples() if fn(ts.metadata["duration_seconds"])]

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
        "male": lambda x: x == " M ",
        "female": lambda x: x == " F ",
    }

    test_cases = []
    for name, fn in test_case_name_to_decision_logic_map.items():
        ts_list = [(ts, gt) for ts, gt in complete_test_case.iter_test_samples() if fn(ts.metadata["speaker_sex"])]

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
        ts_list = [(ts, gt) for ts, gt in complete_test_case.iter_test_samples() if fn(ts.metadata["tempo"])]

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
    df = pd.read_csv(args.dataset_csv, storage_options={"anon": True})
    df = df.where(pd.notnull(df), None)  # read missing cells as None
    df.columns = df.columns.str.replace(r"(\s|\.)+", "_", regex=True)  # sanitize column names to use underscores

    required_columns = {
        "id",
        "locator",
        "text",
        "inference_whisper_default",
        "inference_wav2vec-base-960h",
        "speaker_sex",
        "duration_seconds",
        "max_pitch",
        "energy",
        "zero_crossing_rate",
        "word_count",
        "longest_word_len",
        "tempo",
    }
    assert all(required_column in set(df.columns) for required_column in required_columns)

    non_metadata_columns = {
        "inference_whisper_default",
        "inference_wav2vec-base-960h",
    }

    test_samples = []
    for record in tqdm(df.itertuples(index=False), total=len(df)):
        test_sample = TestSample(
            locator=f"s3://{BUCKET}/{DATASET}/{record.locator}",
            metadata={f: getattr(record, f) for f in required_columns - non_metadata_columns},
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
    kolena.initialize(verbose=True)
    complete_tc = seed_complete_test_case(args)

    test_suite_names: Dict[str, Callable[[str, TestCase], TestSuite]] = {
        f"{DATASET} :: audio duration": seed_test_suite_duration,
        f"{DATASET} :: speaker sex": seed_test_suite_speaker_sex,
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
