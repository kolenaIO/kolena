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
from typing import List

import pandas as pd
from age_estimation.evaluator import evaluate_age_estimation
from age_estimation.workflow import Inference
from age_estimation.workflow import Model
from age_estimation.workflow import TestSample
from age_estimation.workflow import TestSuite

import kolena
from kolena.workflow import test

BUCKET = "kolena-public-datasets"
DATASET = "labeled-faces-in-the-wild"


def seed_test_run(model_name: str, test_suite_names: List[str]) -> None:
    df_results = pd.read_csv(f"s3://{BUCKET}/{DATASET}/predictions/{model_name}.csv")

    def infer(test_sample: TestSample) -> Inference:
        age = df_results[df_results["image_path"] == test_sample.locator]["age"].values[0]
        return Inference(age=age if age != -1 else None)

    model = Model(f"{model_name} [age estimation]", infer=infer)
    print(f"Model: {model}")

    for test_suite_name in test_suite_names:
        test_suite = TestSuite.load(test_suite_name)
        print(f"Test Suite: {test_suite}")

        test(model, test_suite, evaluate_age_estimation)


def main(args: Namespace) -> int:
    kolena.initialize(verbose=True)
    seed_test_run(args.model, args.test_suites)
    return 0


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("model", help="Name of model in directory to test")
    ap.add_argument("test_suites", nargs="+", help="Name(s) of test suite(s) to test.")
    sys.exit(main(ap.parse_args()))
