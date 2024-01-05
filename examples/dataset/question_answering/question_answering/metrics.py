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
from typing import List
from typing import Tuple

import numpy as np
from openai import OpenAI
from question_answering.constants import CONSISTENCY_PROMPT
from sentence_transformers import CrossEncoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"
openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")


def compute_contradiction_score(reference: str, prediction: str) -> Tuple[str, float]:
    """
    The Cross-Encoder for Natural Language Inference (NLI) is a text classification model that takes a pair of text and
    assigns a label: 'contradiction', 'entailment' or 'neutral'. Additionally, it assigns a score, a probability that
    this pair belongs to each label, from 0 to 1. The higher the score, the more confident the model is to assign that
    label.

    To detect hallucination, we use `contradiction` score. The higher the score is, the more likely it is that the
    reference and the prediction are contradicting each other.

    The output is a tuple of NLI label and `contradiction` score.
    """
    nli_scores = nli_model.predict([(reference, prediction)], apply_softmax=True)
    label_mapping = ["contradiction", "entailment", "neutral"]
    nli_label = label_mapping[nli_scores.argmax(axis=1)[0]]
    contradiction_score = nli_scores[0].tolist()[0]

    return nli_label, contradiction_score


def compute_consistency_score(answers: List[str]) -> float:
    """
    The consistency score requires N number of generated answers with the same prompt and it compares each answer with
    the first answer to check for consistency. The more consistent the answers are, the less likely the model is
    hallucinating. The score is computed by prompting GPT-3.5-turbo for each pair of text asking whether it thinks that
    the pair is contradicting or supporting each other. The score is # of consistent answer / total # of answers.

    For example using N = 5 (also this is our recommendation), there are 4 pairs of text to check for consistency and
    often any single inconsistent pair would mean that the model is uncertain about its answer. Therefore, in this case,
    using a threshold of 0.75 (= 3/4) is recommended.
    """
    consistency_scores = []
    for answer in answers[1:]:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": CONSISTENCY_PROMPT,
                },
                {
                    "role": "assistant",
                    "content": "Certainly! Please provide me with the texts for evaluation.",
                },
                {
                    "role": "user",
                    "content": f"Context: {answers[0]}\n\nSentence: {answer}",
                },
            ],
            temperature=0.5,
            max_tokens=50,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        response = str(response.choices[0].message.content)

        consistency_scores.append(response.lower() == "yes")

    return np.mean(consistency_scores)
