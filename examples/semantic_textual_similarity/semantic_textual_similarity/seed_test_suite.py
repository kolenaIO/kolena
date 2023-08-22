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
import os
from argparse import ArgumentParser
from argparse import Namespace

import pandas as pd
from semantic_textual_similarity.workflow import GroundTruth
from semantic_textual_similarity.workflow import SentencePair
from semantic_textual_similarity.workflow import TestCase
from semantic_textual_similarity.workflow import TestSuite
from tqdm import tqdm

import kolena
from kolena.workflow import Text

DATASET = "sts-benchmark"


def seed_complete_test_case(args: Namespace) -> TestCase:
    df = pd.read_csv(args.dataset_csv)
    required_columns = {"sentence1", "sentence2"}
    assert all(required_column in set(df.columns) for required_column in required_columns)

    optional_columns = {"similarity", "cos_similarity", "dot_score"}
    test_samples = []
    for record in tqdm(df.itertuples(index=False), total=len(df)):
        test_sample = SentencePair(  # type: ignore
            sentence1=Text(record.sentence1),
            sentence2=Text(record.sentence2),
            sentence1_word_count=record.sentence1_word_count,
            sentence2_word_count=record.sentence2_word_count,
            sentence1_char_length=record.sentence1_char_length,
            sentence2_char_length=record.sentence2_char_length,
            word_count_diff=record.word_count_diff,
            char_length_diff=record.char_length_diff,
            metadata={f: getattr(record, f) for f in set(record._fields) - required_columns - optional_columns},
        )
        ground_truth = GroundTruth(similarity=record.similarity / 5.0)  # STS-b's similarity ranges from 0 to 5
        test_samples.append((test_sample, ground_truth))

    test_case = TestCase(f"complete :: {DATASET}", test_samples=test_samples, reset=True)
    print(f"Created test case: {test_case}")

    return test_case


def main(args: Namespace) -> None:
    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)
    complete_test_case = seed_complete_test_case(args)
    test_suite = TestSuite(
        f"{DATASET}",
        test_cases=[complete_test_case],
        reset=True,
    )
    print(f"Created test suite: {test_suite}")


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset_csv",
        type=str,
        default="s3://kolena-public-datasets/sts-benchmark/results/all-distilroberta-v1.csv",
        help="CSV file specifying dataset. See default CSV for details",
    )

    main(ap.parse_args())
