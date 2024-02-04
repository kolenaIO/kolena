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
from classification.evaluator import evaluate_classification
from classification.workflow import Inference
from classification.workflow import Model
from classification.workflow import TestSample
from classification.workflow import TestSuite
from classification.workflow import ThresholdConfiguration

import kolena
from kolena.workflow import test
from kolena.workflow.annotation import ScoredClassificationLabel

BUCKET = "kolena-public-datasets"
DATASET = "cifar10/test"


def seed_test_run(model_name: str, test_suite_names: List[str]) -> None:
    df_results = pd.read_csv(f"s3://{BUCKET}/{DATASET}/results/{model_name}.csv", storage_options={"anon": True})

    labels = set(df_results.columns) - {"locator"}

    def infer(test_sample: TestSample) -> Inference:
        sample_result = df_results[df_results["locator"] == test_sample.locator].iloc[0]

        inferences: List[ScoredClassificationLabel] = []
        for label in labels:
            inferences.append(ScoredClassificationLabel(label=label, score=sample_result[label]))

        inferences.sort(key=lambda x: x.score, reverse=True)
        return Inference(inferences=inferences)

    model = Model(f"{model_name} [{DATASET}]", infer=infer)  # type: ignore
    print(f"Model: {model}")

    for test_suite_name in test_suite_names:
        test_suite = TestSuite.load(test_suite_name)
        print(f"Test Suite: {test_suite}")

        test(
            model,
            test_suite,
            evaluate_classification,  # type: ignore
            configurations=[ThresholdConfiguration()],
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
        default=["resnet50v2", "inceptionv3"],
        nargs="+",
        help="Name(s) of model(s) in directory to test",
    )
    ap.add_argument(
        "--test-suites",
        default=[f"image properties :: {DATASET}"],
        nargs="+",
        help="Name(s) of test suite(s) to test.",
    )
    sys.exit(main(ap.parse_args()))
