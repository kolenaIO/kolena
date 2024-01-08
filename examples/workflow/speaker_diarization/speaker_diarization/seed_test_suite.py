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
from argparse import ArgumentParser
from argparse import Namespace
from typing import Callable
from typing import Dict

import pandas as pd
from speaker_diarization.utils import calculate_tertiles
from speaker_diarization.workflow import GroundTruth
from speaker_diarization.workflow import TestCase
from speaker_diarization.workflow import TestSample
from speaker_diarization.workflow import TestSuite
from tqdm import tqdm

import kolena
from kolena.workflow.annotation import LabeledTimeSegment

BUCKET = "kolena-public-datasets"
DATASET = "ICSI-corpus"


def seed_test_suite_by_avg_amp(
    test_suite_name: str,
    complete_test_case: TestCase,
) -> TestSuite:
    test_case_name_to_decision_logic_map = calculate_tertiles(complete_test_case, "Average_Amplitude")
    test_cases = []
    for name, fn in test_case_name_to_decision_logic_map.items():
        ts_list = [
            (ts, gt) for ts, gt in complete_test_case.iter_test_samples() if fn(ts.metadata["Average_Amplitude"])
        ]

        new_tc = TestCase(
            f"average amplitude :: {name} :: {DATASET}",
            test_samples=ts_list,
            reset=True,
        )
        test_cases.append(new_tc)

    test_suite = TestSuite(
        test_suite_name,
        test_cases=[complete_test_case, *test_cases],
        reset=True,
        tags={DATASET, test_suite_name},
    )
    print(f"created test suite {test_suite.name} v{test_suite.version}")


def seed_complete_test_case(args: Namespace) -> TestCase:
    df = pd.read_csv(args.dataset_csv, storage_options={"anon": True})
    df = df.where(pd.notnull(df), None)  # read missing cells as None
    df.columns = df.columns.str.replace(r"(\s|\.)+", "_", regex=True)  # sanitize column names to use underscores

    required_columns = {
        "audio_path",
        "transcription_path",
        "original_audio",
        "audio_length",
        "Average_Amplitude",
        "Zero_Crossing_Rate",
        "Energy",
        "Num_Speakers",
    }
    assert all(required_column in set(df.columns) for required_column in required_columns)

    test_samples = []

    for record in tqdm(df.itertuples(index=False), total=len(df)):
        test_sample = TestSample(
            locator=f"s3://{BUCKET}/{DATASET}/{record.audio_path}",
            metadata={f: getattr(record, f) for f in required_columns},
        )
        transcription_df = pd.read_csv(
            f"s3://{BUCKET}/{DATASET}/{record.transcription_path[:-4] + '_cleaned.csv'}",
            storage_options={"anon": True},
        )
        ground_truth = GroundTruth(
            transcription=[
                LabeledTimeSegment(
                    start=row.starttime,
                    end=row.endtime,
                    label=row.text,
                    group=row.speaker,
                )
                for idx, row in transcription_df.iterrows()
            ],
        )
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
        f"{args.test_suite} :: average amplitude": seed_test_suite_by_avg_amp,
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

    ap.add_argument(
        "--test_suite",
        type=str,
        default=DATASET,
        help="Optionally specify a name for the created test suites.",
    )

    main(ap.parse_args())
