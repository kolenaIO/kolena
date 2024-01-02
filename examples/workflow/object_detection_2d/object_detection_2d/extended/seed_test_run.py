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

import pandas as pd
from object_detection_2d.constants import MODEL_METADATA
from object_detection_2d.extended.workflow import Model
from object_detection_2d.extended.workflow import TestSuite
from object_detection_2d.seed_test_run import get_stored_inferences
from object_detection_2d.seed_test_run import load_results
from object_detection_2d.seed_test_run import MODEL_LIST
from object_detection_2d.seed_test_run import setup_evaluator
from object_detection_2d.seed_test_run import TEST_SUITE_NAMES

import kolena
from kolena.workflow.test_run import test


def seed_test_run(
    model_alias: str,
    model_full_name: str,
    test_suite_name: str,
    groups_df: pd.DataFrame,
) -> None:
    # create a model
    model = Model(model_full_name, infer=get_stored_inferences(groups_df), metadata=MODEL_METADATA[model_alias])

    # customizable configurations for the evaluator
    evaluator = setup_evaluator()

    # runs the evaluation
    test_suite = TestSuite(test_suite_name)
    test(model, test_suite, evaluator, reset=True)


def main(args: Namespace) -> None:
    model_alias = args.model
    model_full_name = MODEL_LIST[model_alias]

    # run evaluation on test suites
    kolena.initialize(verbose=True)

    metadata_by_image = load_results(model_alias)

    if args.test_suite == "none":
        for name in TEST_SUITE_NAMES:
            seed_test_run(model_alias, model_full_name, name, metadata_by_image)
    else:
        seed_test_run(model_alias, model_full_name, args.test_suite, metadata_by_image)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("model", choices=MODEL_LIST.keys(), help="The alias of the model to test.")
    ap.add_argument(
        "--test-suite",
        type=str,
        default="none",
        help="Optionally specify a test suite to test. Test against all available test suites when unspecified.",
    )

    main(ap.parse_args())
