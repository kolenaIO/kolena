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
from typing import Any
from typing import Dict

import evaluate
import pandas as pd

bertscore = evaluate.load("bertscore")
bleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")


def compute_metrics(gt: str, inf: str) -> Dict[str, Any]:
    gt_word_count = len(gt.split()) if not pd.isna(gt) else 0
    inf_word_count = len(inf.split()) if not pd.isna(inf) else 0
    if pd.isna(gt) or inf_word_count <= 3:
        return dict(
            is_failure=True,
            inference_word_count=inf_word_count,
            BERT_precision=0.0,
            BERT_recall=0.0,
            BERT_f1=0.0,
            ROUGE_1=0.0,
            ROUGE_2=0.0,
            ROUGE_L=0.0,
            BLEU=0.0,
            inf_to_gt_word_count=float(inf_word_count) / gt_word_count if gt_word_count != 0 else 0.0,
        )

    bertscore_results = bertscore.compute(
        predictions=[inf],
        references=[gt],
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli",
    )
    bleu_results = bleu.compute(predictions=[inf], references=[gt])
    rouge_results = rouge.compute(
        predictions=[inf],
        references=[gt],
        rouge_types=["rouge1", "rouge2", "rougeL"],
    )

    return dict(
        is_failure=False,
        inference_word_count=inf_word_count,
        BERT_precision=bertscore_results["precision"][0],
        BERT_recall=bertscore_results["recall"][0],
        BERT_f1=bertscore_results["f1"][0],
        ROUGE_1=rouge_results["rouge1"],
        ROUGE_2=rouge_results["rouge2"],
        ROUGE_L=rouge_results["rougeL"],
        BLEU=bleu_results["score"] / 100.0,
        inf_to_gt_word_count=float(inf_word_count) / gt_word_count,
    )
