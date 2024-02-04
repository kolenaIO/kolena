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
from typing import List

import pandas as pd
from semantic_textual_similarity.evaluator import evaluate_semantic_similarity
from semantic_textual_similarity.workflow import Inference
from semantic_textual_similarity.workflow import Model
from semantic_textual_similarity.workflow import SentencePair
from semantic_textual_similarity.workflow import TestSuite

import kolena
from kolena.workflow.test_run import test

BUCKET = "kolena-public-datasets"
DATASET = "sts-benchmark"


def seed_test_run(model_name: str, test_suite_names: List[str]) -> None:
    df = pd.read_csv(f"s3://{BUCKET}/{DATASET}/results/{model_name}.csv", storage_options={"anon": True})

    required_columns = {"sentence1", "sentence2", "cos_similarity"}
    assert all(required_column in set(df.columns) for required_column in required_columns)

    def infer(test_sample: SentencePair) -> Inference:
        sample_result = df[
            (df["sentence1"] == test_sample.sentence1.text) & (df["sentence2"] == test_sample.sentence2.text)
        ].iloc[0]

        return Inference(similarity=sample_result["cos_similarity"])

    model = Model(f"{model_name}", infer=infer)  # type: ignore
    print(f"Model: {model}")

    for test_suite_name in test_suite_names:
        test_suite = TestSuite.load(test_suite_name)
        print(f"Test Suite: {test_suite}")

        test(
            model,
            test_suite,
            evaluate_semantic_similarity,
            reset=True,
        )


def main(args: Namespace) -> int:
    kolena.initialize(verbose=True)
    for model_name in args.models:
        seed_test_run(model_name, args.test_suites)
    return 0


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--models",
        default=["all-distilroberta-v1", "all-MiniLM-L12-v2", "all-mpnet-base-v2"],
        nargs="+",
        help="Name(s) of model(s) in directory to test",
    )
    ap.add_argument(
        "--test-suites",
        default=[DATASET],
        nargs="+",
        help="Name(s) of test suite(s) to test.",
    )
    sys.exit(main(ap.parse_args()))
