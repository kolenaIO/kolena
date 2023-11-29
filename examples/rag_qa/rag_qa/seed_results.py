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
from typing import Optional

from rag_qa.constants import HALU_DIALOG
from rag_qa.constants import HALU_QA
from rag_qa.constants import HALU_SUMMARIZATION
from rag_qa.constants import SQUAD2_DEV
from rag_qa.data_loader import HALU_MODELS
from rag_qa.data_loader import load_halu_dialog_results
from rag_qa.data_loader import load_halu_qa_results
from rag_qa.data_loader import load_halu_summarization_results
from rag_qa.data_loader import load_squad_dev_results
from rag_qa.data_loader import SQUAD_MODELS

import kolena
from kolena._experimental.dataset import test


def submit_squad_dev(model: str, dataset: Optional[str] = None) -> None:
    dataset = dataset or "SQuAD 2.0 Dev"
    result = load_squad_dev_results(model)
    test(dataset, model, result, on="id")


def submit_qa(model: str, dataset: Optional[str] = None) -> None:
    dataset = dataset or "HaLuEval qa"
    qa_results, config = load_halu_qa_results(model)

    test(dataset, model, [(config, qa_results.rename(columns={"knowledge": "text"}))], on=["text", "question"])
    print(qa_results.shape)


def submit_dialogue(model: str, dataset: Optional[str] = None) -> None:
    dataset = dataset or "HaLuEval dialogue"
    dialogue_results, config = load_halu_dialog_results(model)

    test(dataset, model, [(config, dialogue_results)], on=["text", "dialogue_history"])
    print(dialogue_results.shape)


def submit_summarization(model: str, dataset: Optional[str] = None) -> None:
    dataset = dataset or "HaLuEval summarization"
    summarization_results, config = load_halu_summarization_results(model)

    # TODO: this does not work yet because of non-trivial join between data and result json
    test(dataset, model, [(config, summarization_results)], on=["text"])
    print(summarization_results.shape)


proc = {
    "squad2-dev": submit_squad_dev,
    "halu-qa": submit_qa,
    "halu-dialog": submit_dialogue,
    "halu-summarization": submit_summarization,
}


def main(args: Namespace) -> None:
    kolena.initialize(verbose=True)

    proc[args.benchmark](args.model, args.dataset_name)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--benchmark",
        choices=[SQUAD2_DEV, HALU_QA, HALU_DIALOG, HALU_SUMMARIZATION],
        required=True,
        help="Name of the benchmark to seed.",
    )
    ap.add_argument(
        "--model",
        choices=SQUAD_MODELS + HALU_MODELS,
        required=True,
        help="Name of the model to seed.",
    )
    ap.add_argument("--dataset-name", help="dataset name")

    main(ap.parse_args())
