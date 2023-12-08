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
import sys
from argparse import ArgumentParser
from argparse import Namespace

import pandas as pd
from question_answering.truthful_qa.workflow import GroundTruth
from question_answering.truthful_qa.workflow import TestCase
from question_answering.truthful_qa.workflow import TestSample
from question_answering.truthful_qa.workflow import TestSuite

import kolena


BUCKET = "kolena-public-datasets"
DATASET = "TruthfulQA"


def main(args: Namespace) -> int:
    kolena.initialize(verbose=True)

    df_datapoints = pd.read_csv(args.dataset_csv)

    non_metadata_fields = {
        "Question",
        "Best Answer",
        "Correct Answers",
        "Incorrect Answers",
    }

    # Create a list of every test sample with its ground truth
    test_samples_and_ground_truths = [
        (
            TestSample(
                text=f"Q:\n{row['Question']}\n\nA: \n{row['Best Answer']}",
                question=str(row["Question"]),
                context=[context for context in str(row["Context"]).split("\n") if context != ""],
                metadata={key.lower(): row[key] for key in set(row.keys()) - non_metadata_fields},
            ),
            GroundTruth(
                best_answer=str(row["Best Answer"]),
                answers=[answer for answer in str(row["Correct Answers"]).split(";")],
                incorrect_answers=[answer for answer in str(row["Incorrect Answers"]).split(";")],
            ),
        )
        for _, row in df_datapoints.iterrows()
    ]

    complete_test_case = TestCase(
        f"complete {DATASET}",
        description=f"All questions and answers in the {DATASET} dataset",
        test_samples=test_samples_and_ground_truths,
        reset=True,
    )

    test_cases = []

    test_cases.append(
        TestCase(
            f"adversarial :: {DATASET}",
            test_samples=[
                (ts, gt) for ts, gt in test_samples_and_ground_truths if ts.metadata["type"] == "Adversarial"
            ],
            reset=True,
        ),
    )

    test_cases.append(
        TestCase(
            f"non-adversarial :: {DATASET}",
            test_samples=[
                (ts, gt) for ts, gt in test_samples_and_ground_truths if ts.metadata["type"] == "Non-Adversarial"
            ],
            reset=True,
        ),
    )

    TestSuite(DATASET, test_cases=[complete_test_case, *test_cases], reset=True)

    return 0


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset_csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/TruthfulQA_QA-context.csv",
        help="CSV file with questions and answers.",
    )
    sys.exit(main(ap.parse_args()))
