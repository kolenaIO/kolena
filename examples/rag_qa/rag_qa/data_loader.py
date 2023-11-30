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
from typing import Any
from typing import Dict
from typing import Tuple

import pandas as pd
from rag_qa.constants import BUCKET
from rag_qa.evaluation_prompts import dialogue_instruction
from rag_qa.evaluation_prompts import qa_instruction
from rag_qa.evaluation_prompts import summarization_instruction

from kolena.workflow.annotation import Utterance

SQUAD_MODELS = ["IE-Net (ensemble)", "FPNet (ensemble)"]
HALU_MODELS = ["gpt-3.5-turbo"]


def load_squad2_dev(sample_count: int = 0) -> pd.DataFrame:
    dev_json = pd.read_json(f"s3://{BUCKET}/SQuAD2/dev-v2.0.json")

    dev = pd.DataFrame(
        [
            {
                "title": r["title"],
                "context": p["context"],
                "question": qas["question"],
                "id": qas["id"],
                "answers": [Utterance(text=answer["text"], start=answer["answer_start"]) for answer in qas["answers"]],
                "is_impossible": qas["is_impossible"],
                "plausible_answers": [
                    Utterance(text=answer["text"], start=answer["answer_start"])
                    for answer in qas.get("plausible_answers", [])
                ],
            }
            for r in dev_json["data"]
            for p in r["paragraphs"]
            for qas in p["qas"]
        ],
    )

    dev["text"] = dev[["context", "question"]].apply(
        lambda x: "Context:\n{}\n\nQuestion\n{}".format(x["context"], x["question"]),
        axis=1,
    )
    return dev[:sample_count] if sample_count else dev


def load_squad2_dev_results(model: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    assert model in SQUAD_MODELS
    result = (
        pd.read_json(f"s3://{BUCKET}/SQuAD2/results/{model}.json", orient="index")
        .reset_index(
            names="id",
        )
        .rename(columns={0: "answer"})
    )
    return result, {}


def load_squad2_train(sample_count: int = 0) -> pd.DataFrame:
    train_json = pd.read_json(f"s3://{BUCKET}/SQuAD2/train-v2.0.json")

    train = pd.DataFrame(
        [
            {
                "title": r["title"],
                "context": p["context"],
                "question": qas["question"],
                "id": qas["id"],
                "answers": [Utterance(text=answer["text"], start=answer["answer_start"]) for answer in qas["answers"]],
                "is_impossible": qas["is_impossible"],
                "plausible_answers": [
                    Utterance(text=answer["text"], start=answer["answer_start"])
                    for answer in qas.get(
                        "plausible_answers",
                        [],
                    )
                ],
            }
            for r in train_json["data"]
            for p in r["paragraphs"]
            for qas in p["qas"]
        ],
    )

    train["text"] = train[["context", "question"]].apply(
        lambda x: "Context:\n{}\n\nQuestion\n{}".format(
            x["context"],
            x["question"],
        ),
        axis=1,
    )
    return train[:sample_count] if sample_count else train


def load_halu_qa(sample_count: int = 0) -> pd.DataFrame:
    qa = pd.read_json(f"s3://{BUCKET}/HaLuEval/data/qa_data.json", lines=True).rename(columns={"knowledge": "text"})
    return qa[:sample_count] if sample_count else qa


def load_halu_qa_results(model: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    assert model in HALU_MODELS
    qa_results = pd.read_json(f"s3://{BUCKET}/HaLuEval/evaluation/qa_{model}_result.json", lines=True).rename(
        columns={"knowledge": "text"},
    )

    system_prompt = (
        "You are a hallucination detector. You MUST determine if the provided answer contains "
        "hallucination or not for the question based on the world knowledge. The answer you provided "
        'MUST be "Yes" or "No"'
    )
    user_prompt = qa_instruction + "\n\n#Question#: <question>" + "\n#Answer#: <answer>" + "\n#Your Judgement#: "

    return (
        qa_results,
        dict(system_promt=system_prompt, user_prompt=user_prompt),
    )


def load_halu_dialog(sample_count: int = 0) -> pd.DataFrame:
    dialogue = pd.read_json(f"s3://{BUCKET}/HaLuEval/data/dialogue_data.json", lines=True).rename(
        columns={"knowledge": "text"},
    )
    return dialogue[:sample_count] if sample_count else dialogue


def load_halu_dialog_results(model: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    assert model in HALU_MODELS
    dialogue_results = pd.read_json(
        f"s3://{BUCKET}/HaLuEval/evaluation/dialogue_{model}_results.json",
        lines=True,
    ).rename(columns={"knowledge": "text"})

    system_prompt = (
        "You are a response judge. You MUST determine if the provided response contains non-factual or "
        'hallucinated information. The answer you give MUST be "Yes" or "No"'
    )
    user_prompt = (
        dialogue_instruction + "\n\n#Dialogue History#: <dialog>" + "\n#Response#: <response>\n#Your Judgement#: "
    )

    return dialogue_results, dict(system_prompt=system_prompt, user_prompt=user_prompt)


def load_halu_summarization(sample_count: int = 0) -> pd.DataFrame:
    summarization = pd.read_json(f"s3://{BUCKET}/HaLuEval/data/summarization_data.json", lines=True).rename(
        columns={"document": "text"},
    )
    return summarization[:sample_count] if sample_count else summarization


def load_halu_summarization_results(model: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    assert model in HALU_MODELS
    summarization_results = pd.read_json(
        f"s3://{BUCKET}/HaLuEval/evaluation/summarization_{model}_results.json",
        lines=True,
    ).rename(columns={"document": "text"})

    system_prompt = (
        "You are a summary judge. You MUST determine if the provided summary contains non-factual "
        'or hallucinated information. The answer you give MUST be "Yes" or "No"'
    )
    user_prompt = (
        summarization_instruction + "\n\n#Document#: <document>" + "\n#Summary#: <summary>" + "\n#Your Judgement#: "
    )

    return summarization_results, dict(system_promt=system_prompt, user_prompt=user_prompt)
