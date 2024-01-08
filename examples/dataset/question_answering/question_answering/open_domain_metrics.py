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
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from openai import OpenAI
from question_answering.constants import OPEN_DOMAIN_GPT4_HALLUCINATION_PROMPT
from question_answering.metrics import compute_consistency_score
from question_answering.metrics import compute_contradiction_score

openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def compute_open_domain_metrics(question: str, reference: str, prediction: str, answers: List[str]) -> Dict[str, Any]:
    gpt4_hallucination_flag, reason = compute_gpt4_hallucination_flag(question, reference, prediction)
    nli_label, contradiction_score = compute_contradiction_score(reference, prediction)
    metrics = dict(
        gpt4_hallucination_flag=gpt4_hallucination_flag,
        gpt4_hallucination_flag_reason=reason,
        gpt4_hallucination_score=compute_gpt4_hallucination_score(question, reference, prediction),
        nli_label=nli_label,
        contradiction_score=contradiction_score,
        consistency_score=compute_consistency_score(answers),
    )
    return metrics


def compute_gpt4_hallucination_flag(question: str, reference: str, prediction: str) -> Tuple[bool, str]:
    """
    Prompts GPT-4 for hallucination evaluation in a binary format. The output is a `(bool, str)` format where
    the first item in the tuple is a hallucination flag in boolean and the second item is a reason why GPT-4 thinks
    that the predicted answer is a hallucination.

    The output is either `(True, "with explanation")` or `(False, "")`.
    """
    response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "user",
                "content": OPEN_DOMAIN_GPT4_HALLUCINATION_PROMPT,
            },
            {
                "role": "assistant",
                "content": "Certainly! Please provide me with the texts for evaluation.",
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nPerfect Answer: {reference}" f"\n\nGenerated Answer: {prediction}",
            },
        ],
        temperature=0.5,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    decision = str(response.choices[0].message.content)
    if "\n" in decision:
        return decision.split("\n")[0].lower() == "yes", decision.split("\n")[1]

    return False, ""


def compute_gpt4_hallucination_score(question: str, reference: str, prediction: str, repeat: int = 5) -> float:
    """
    Prompts GPT-4 for hallucination evaluation `repeat` number of times. By default it repeats 5 times. The
    hallucination score is computed as an average # of times hallucination is detected.

    The output ranges from 0 to 1 where 0 means no hallucination (0 ouf of 5 times LLM detected hallucination) and 1
    means definitely a hallucination (5 out of 5 times LLM detected hallucination).
    """
    results = [compute_gpt4_hallucination_flag(question, reference, prediction) for _ in range(repeat)]

    hallucation_score = sum([is_hallucination for is_hallucination, _ in results]) / len(results)
    return hallucation_score
