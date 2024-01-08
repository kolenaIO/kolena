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
import sys
from argparse import ArgumentParser
from argparse import Namespace
from collections import defaultdict
from typing import List
from typing import Tuple

import pandas as pd
from question_answering.utils import normalize_string
from question_answering.workflow import GroundTruth
from question_answering.workflow import TestCase
from question_answering.workflow import TestSample
from question_answering.workflow import TestSuite

import kolena
from kolena.workflow.annotation import ClassificationLabel

BUCKET = "kolena-public-datasets"
DATASET = "CoQA"


def clean_name(name: str) -> str:
    return name[9:] if name.startswith("metadata_") else name


def create_test_suite_by_question(
    suite_name: str,
    dataset: TestCase,
    data: List[Tuple[TestSample, GroundTruth]],
) -> None:
    question_types = ["what", "who", "how", "did", "where", "was", "when", "is", "why", "other"]
    samples_by_question_type = {value: [] for value in question_types}

    # Organize samples into the dictionary
    for ts, gt in data:
        question_type = ts.metadata["type_of_question"]
        if question_type in samples_by_question_type:
            samples_by_question_type[question_type].append((ts, gt))
        else:
            samples_by_question_type["other"].append((ts, gt))

    # Create test cases using the organized dictionary
    test_cases = TestCase.init_many(
        data=[(f"question type :: {name} :: {DATASET}", samples_by_question_type[name]) for name in question_types],
        reset=True,
    )

    test_suite = TestSuite(
        f"question types :: {suite_name}",
        test_cases=[dataset, *test_cases],
        reset=True,
    )
    print(f"created test suite: {test_suite.name} v{test_suite.version}")


def create_test_suite_by_conversation_length(
    suite_name: str,
    dataset: TestCase,
    data: List[Tuple[TestSample, GroundTruth]],
) -> None:
    depths = sorted({test_sample.turn for test_sample, _ in data})
    samples_by_turn = {value: [] for value in depths}

    # Organize samples into the dictionary
    for ts, gt in data:
        samples_by_turn[ts.turn].append((ts, gt))

    # Create test cases using the organized dictionary
    test_cases = TestCase.init_many(
        data=[
            (f"conversation depth :: {number:02d} :: {DATASET}", samples_by_turn[number])
            for number in depths
            if len(samples_by_turn[number]) >= 10  # ignoring test cases with fewer than 10 test samples
        ],
        reset=True,
    )

    test_suite = TestSuite(
        f"conversation depths :: {suite_name}",
        test_cases=[dataset, *test_cases],
        reset=True,
    )
    print(f"created test suite: {test_suite.name} v{test_suite.version}")


def main(args: Namespace) -> int:
    kolena.initialize(verbose=True)

    df_metadata = pd.read_csv(args.dataset_csv, storage_options={"anon": True})
    context_dict = defaultdict(list)

    # Store the conversation context for each story (data_id)
    for _, row in df_metadata.iterrows():
        data_id = row["data_id"]
        question = row["question"]
        answer = row["metadata_answer"]
        turn = row["turn"]
        entry = f"{turn:02d} - Q: {question} - A: {answer}"
        context_dict[data_id].append(entry)

    non_metadata_fields = {
        "story",
        "question",
        "data_id",
        "turn",
        "metadata_answer",
        "other_answer_1",
        "other_answer_2",
        "other_answer_3",
    }

    # Create a list of every test sample with its ground truth
    test_samples_and_ground_truths = [
        (
            TestSample(
                data_id=str(record.data_id),
                text=str(record.story),
                question=str(record.question),
                turn=int(record.turn),
                metadata={clean_name(f): getattr(record, f) for f in set(record._fields) - non_metadata_fields},
            ),
            GroundTruth(
                answer=str(record.metadata_answer),
                clean_answer=normalize_string(str(record.metadata_answer)),
                context=context_dict[str(record.data_id)][: record.turn],
                other_answer_1=record.other_answer_1,
                other_answer_2=record.other_answer_2,
                other_answer_3=record.other_answer_3,
                question_answer=ClassificationLabel(label=context_dict[str(record.data_id)][record.turn - 1]),
            ),
        )
        for record in df_metadata.itertuples(index=False)
    ]

    complete_test_case = TestCase(
        f"complete {DATASET}",
        description=f"All questions and answers in the {DATASET} dataset",
        test_samples=test_samples_and_ground_truths,
        reset=True,
    )

    suite_name = args.test_suite
    create_test_suite_by_question(suite_name, complete_test_case, test_samples_and_ground_truths)
    create_test_suite_by_conversation_length(suite_name, complete_test_case, test_samples_and_ground_truths)

    return 0


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset_csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/metadata/metadata.csv",
        help="CSV file with a stories, questions, and answers.",
    )
    ap.add_argument(
        "--test_suite",
        type=str,
        default=DATASET,
        help="Optionally specify a name for the created test suites.",
    )
    sys.exit(main(ap.parse_args()))
