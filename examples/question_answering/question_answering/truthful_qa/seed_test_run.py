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
import ast
import sys
from argparse import ArgumentParser
from argparse import Namespace
from collections import defaultdict

import pandas as pd
from question_answering.truthful_qa.evaluator import evaluate_question_answering
from question_answering.truthful_qa.workflow import Answer
from question_answering.truthful_qa.workflow import Inference
from question_answering.truthful_qa.workflow import Model
from question_answering.truthful_qa.workflow import TestSample
from question_answering.truthful_qa.workflow import TestSuite

import kolena
from kolena.workflow import test
from kolena.workflow.annotation import Label
from kolena.workflow.asset import PlainTextAsset


MODELS = ["gpt-3.5-turbo-instruct"]
BUCKET = "kolena-public-datasets"
DATASET = "TruthfulQA"


def main(args: Namespace) -> int:
    kolena.initialize(verbose=True)

    inference_map = defaultdict(list)
    inference_map_top5 = {}
    model_file_path = f"s3://{BUCKET}/{DATASET}/results/{args.model}_labeled.csv"
    model_file_with_top5_path = f"s3://{BUCKET}/{DATASET}/results/{args.model}_top5.csv"
    df = pd.read_csv(model_file_path)
    df = df.dropna()
    df_halu = df[["question", "is_hallucination"]]
    df_top5 = pd.read_csv(model_file_with_top5_path)
    df_top5 = df_top5.dropna()
    df_top5 = pd.merge(df_top5, df_halu, on="question", how="left")

    for _, row in df.iterrows():
        inference_map[row["question"]].append(
            (
                Answer(
                    label=row["answer"],
                    raw=row["raw_answer"],
                    text_offset=ast.literal_eval(row["tokens_text_offset"]),
                    logprobs=ast.literal_eval(row["tokens_logprobs"]),
                    tokens=ast.literal_eval(row["tokens"]),
                    top_logprobs=ast.literal_eval(row["tokens_top_logprobs"]),
                    finish_reason=row["finish_reason"],
                    completion_tokens=ast.literal_eval(str(row["completion_tokens"])),
                    prompt_tokens=ast.literal_eval(str(row["prompt_tokens"])),
                    total_tokens=ast.literal_eval(str(row["total_tokens"])),
                ),
                row["is_hallucination"],
            ),
        )

    for _, row in df_top5.iterrows():
        inference_map_top5[row["question"]] = Answer(
            label=row["answer"],
            raw=row["raw_answer"],
            text_offset=ast.literal_eval(row["tokens_text_offset"]),
            logprobs=ast.literal_eval(row["tokens_logprobs"]),
            tokens=ast.literal_eval(row["tokens"]),
            top_logprobs=ast.literal_eval(row["tokens_top_logprobs"]),
            finish_reason=row["finish_reason"],
            completion_tokens=ast.literal_eval(str(row["completion_tokens"])),
            prompt_tokens=ast.literal_eval(str(row["prompt_tokens"])),
            total_tokens=ast.literal_eval(str(row["total_tokens"])),
        )

    inference_map_x5 = defaultdict(list)
    model_file_path = f"s3://{BUCKET}/{DATASET}/results/{args.model}x5.csv"
    df = pd.read_csv(model_file_path)
    df = df.dropna()

    for _, row in df.iterrows():
        for i in range(5):
            inference_map_x5[row["question"]].append(row[i + 1])

    # define a function that generates an inference from a test sample
    def infer(test_sample: TestSample) -> Inference:
        if len(inference_map[test_sample.question]) == 0:
            return Inference(
                missing_answer=len(inference_map[test_sample.question]) == 0,
                answers=[],
            )

        return Inference(
            missing_answer=len(inference_map[test_sample.question]) == 0,
            answers=[Label(label=ans) for ans in inference_map_x5[test_sample.question]],
            answer=Label(label=inference_map[test_sample.question][0][0].label),
            answer_with_top5_logprob=Label(label=inference_map_top5[test_sample.question].label),
            selfcheck_metrics=PlainTextAsset(
                locator=f"s3://{BUCKET}/{DATASET}/results/selfcheck/{args.model}x5_bertscore_ngram_metrics.csv",
            ),
            probabilities_metrics=PlainTextAsset(
                locator=f"s3://{BUCKET}/{DATASET}/results/prob-based_uncertainty/{args.model}_metrics.csv",
            ),
            consistency_metrics=PlainTextAsset(
                locator=f"s3://{BUCKET}/{DATASET}/results/selfcheck/{args.model}x5_consistency_metrics.csv",
            ),
            is_hallucination=inference_map[test_sample.question][0][1]
            if inference_map[test_sample.question][0][1] != "inconclusive"
            else None,
        )

    model = Model(args.model, infer=infer)
    test_suite = TestSuite.load(args.test_suite)
    test(model, test_suite, evaluate_question_answering, reset=True)
    return 0


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--test_suite", default=DATASET, help="Name of the test suite to test.")
    ap.add_argument("--model", default="gpt-3.5-turbo-instruct", choices=MODELS, help="Name of the model to test.")

    sys.exit(main(ap.parse_args()))
