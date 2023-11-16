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
from rag_qa.seed_dataset import BUCKET

import kolena
from kolena._experimental.dataset import test


def submit_squad_dev() -> None:
    model = "IE-Net (ensemble)"
    # model = "FPNet (ensemble)"
    result_json = pd.read_json(f"s3://{BUCKET}/SQuAD2/results/{model}.json")
    result = pd.DataFrame([dict(id=id, answer=ans) for id, ans in result_json.items()])
    test("SQuAD 2.0 Dev", model, result, on="id")


def submit_qa() -> None:
    qa_results = pd.read_json(f"s3://{BUCKET}/HaLuEval/evaluation/qa/qa_gpt-3.5-turbo_result.json", lines=True)
    kolena.initialize()

    with open("evaluation/qa/qa_evaluation_instruction.txt") as f:
        instruction = f.read()

    system_prompt = (
        "You are a hallucination detector. You MUST determine if the provided answer contains "
        "hallucination or not for the question based on the world knowledge. The answer you provided "
        'MUST be "Yes" or "No"'
    )
    user_prompt = instruction + "\n\n#Question#: <question>" + "\n#Answer#: <answer>" + "\n#Your Judgement#: "

    test(
        "HaLuEval qa",
        "gpt-3.5-turbo",
        [
            (
                dict(system_promt=system_prompt, user_prompt=user_prompt),
                qa_results.rename(columns={"knowledge": "text"}),
            ),
        ],
        on=["text", "question"],
    )
    print(qa_results.shape)


def submit_dialogue() -> None:
    # dialogue = pd.read_json("data/dialogue_data.json", lines=True)
    dialogue_results = pd.read_json(
        f"s3://{BUCKET}/HaLuEval/evaluation/dialogue/dialogue_gpt-3.5-turbo_results.json",
        lines=True,
    )

    with open("evaluation/dialogue/dialogue_evaluation_instruction.txt") as f:
        instruction = f.read()

    system_prompt = (
        "You are a response judge. You MUST determine if the provided response contains non-factual or "
        'hallucinated information. The answer you give MUST be "Yes" or "No"'
    )
    user_prompt = instruction + "\n\n#Dialogue History#: <dialog>" + "\n#Response#: <response>" + "\n#Your Judgement#: "

    kolena.initialize()
    test(
        "HaLuEval dialogue",
        "gpt-3.5-turbo",
        [
            (
                dict(system_prompt=system_prompt, user_prompt=user_prompt),
                dialogue_results.rename(columns={"knowledge": "text"}),
            ),
        ],
        on=["text", "dialogue_history"],
    )
    print(dialogue_results.shape)


def submit_summarization() -> None:
    summarization_results = pd.read_json(
        "evaluation/summarization/summarization_gpt-3.5-turbo_results.json",
        lines=True,
    )

    with open("evaluation/summarization/summarization_evaluation_instruction.txt") as f:
        instruction = f.read()

    system_prompt = (
        "You are a summary judge. You MUST determine if the provided summary contains non-factual "
        'or hallucinated information. The answer you give MUST be "Yes" or "No"'
    )
    user_prompt = instruction + "\n\n#Document#: <document>" + "\n#Summary#: <summary>" + "\n#Your Judgement#: "

    test(
        "HaLuEval summarization",
        "gpt-3.5-turbo",
        [
            (
                dict(system_promt=system_prompt, user_prompt=user_prompt),
                summarization_results.rename(columns={"document": "text"}),
            ),
        ],
        on=["text"],
    )
    print(summarization_results.shape)


def main(args: Namespace) -> None:
    kolena.initialize()


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--benchmark",
        choices=["squad2-dev", "squad2-train", "halu-qa", "halu-dialog", "halu-summarization"],
        required=True,
        help="Name of the benchmark to seed.",
    )
    ap.add_argument(
        "--model",
        choices=["squad2-dev", "squad2-train", "halu-qa", "halu-dialog", "halu-summarization"],
        required=True,
        help="Name of the benchmark to seed.",
    )
    ap.add_argument("--dataset-name", help="")

    main(ap.parse_args())
