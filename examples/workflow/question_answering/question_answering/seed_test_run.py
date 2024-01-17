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
from typing import Any
from typing import Dict

import pandas as pd
from question_answering.evaluator import evaluate_question_answering
from question_answering.utils import normalize_string
from question_answering.workflow import Inference
from question_answering.workflow import Model
from question_answering.workflow import TestSample
from question_answering.workflow import TestSuite
from question_answering.workflow import ThresholdConfiguration

import kolena
from kolena.workflow import test
from kolena.workflow.annotation import ClassificationLabel

MODELS = ["gpt-3.5-turbo-0301", "gpt-3.5-turbo", "gpt-4-0314", "gpt-4"]


def main(args: Namespace) -> int:
    kolena.initialize(verbose=True)

    inference_mapping: Dict[Any, Dict[Any, Dict[str, Any]]] = {}
    model_file_path = f"s3://kolena-public-datasets/CoQA/results/{args.model}.csv"
    df = pd.read_csv(model_file_path, storage_options={"anon": True})

    # populate inference_mapping
    for _, row in df.iterrows():
        data_id = row["data_id"]
        turn = row["turn"]
        inference_mapping.setdefault(data_id, {})[turn] = {
            "answer": row["answer"],
            "wc_answer": row["wc_answer"],
            "inference_prompt_tokens": row["inference_prompt_tokens"],
            "inference_completion_tokens": row["inference_completion_tokens"],
            "inference_total_tokens": row["inference_total_tokens"],
        }

    # define a function that generates an inference from a test sample
    def infer(test_sample: TestSample) -> Inference:
        result = inference_mapping[test_sample.data_id][test_sample.turn]
        return Inference(
            answer=ClassificationLabel(label=result["answer"]),
            clean_answer=normalize_string(result["answer"]),
            wc_answer=result["wc_answer"],
            inference_prompt_tokens=result["inference_prompt_tokens"],
            inference_completion_tokens=result["inference_completion_tokens"],
            inference_total_tokens=result["inference_total_tokens"],
            source=model_file_path,
        )

    # define configurations
    configurations = [
        ThresholdConfiguration(threshold=0.5),
    ]

    model = Model(args.model, infer=infer)  # type: ignore
    test_suite = TestSuite.load(args.test_suite)
    test(model, test_suite, evaluate_question_answering, configurations, reset=True)  # type: ignore
    return 0


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--test-suite", default="question types :: CoQA", help="Name of the test suite to test.")
    ap.add_argument("--model", default="gpt-4", choices=MODELS, help="Name of the model to test.")

    sys.exit(main(ap.parse_args()))
