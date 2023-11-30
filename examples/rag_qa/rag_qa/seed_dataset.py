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
from collections import OrderedDict

from rag_qa.constants import HALU_DIALOG
from rag_qa.constants import HALU_QA
from rag_qa.constants import HALU_SUMMARIZATION
from rag_qa.constants import SQUAD2_DEV
from rag_qa.constants import SQUAD2_TRAIN
from rag_qa.data_loader import load_halu_dialog
from rag_qa.data_loader import load_halu_qa
from rag_qa.data_loader import load_halu_summarization
from rag_qa.data_loader import load_squad2_dev
from rag_qa.data_loader import load_squad2_train

import kolena
from kolena._experimental.dataset import register_dataset


def seed_squad2_dev(sample_count: int = 0):
    dataset = "SQuAD 2.0 Dev"
    if sample_count:
        dataset = f"{dataset} ({sample_count})"
    register_dataset(dataset, load_squad2_dev(sample_count))


def seed_squad2_train(sample_count: int = 0):
    dataset = "SQuAD 2.0 Train"
    if sample_count:
        dataset = f"{dataset} ({sample_count})"
    register_dataset(dataset, load_squad2_train(sample_count))


def seed_halu_qa(sample_count: int = 0):
    dataset = "HaLuEval qa"
    if sample_count:
        dataset = f"{dataset} ({sample_count})"
    register_dataset(dataset, load_halu_qa())


def seed_halu_dialog(sample_count: int = 0):
    dataset = "HaLuEval dialogue"
    if sample_count:
        dataset = f"{dataset} ({sample_count})"
    register_dataset(dataset, load_halu_dialog())


def seed_halu_summarization(sample_count: int = 0):
    dataset = "HaLuEval summarization"
    if sample_count:
        dataset = f"{dataset} ({sample_count})"
    register_dataset(dataset, load_halu_summarization())


proc = OrderedDict(
    [
        (SQUAD2_DEV, seed_squad2_dev),
        (SQUAD2_TRAIN, seed_squad2_train),
        (HALU_QA, seed_halu_qa),
        (HALU_DIALOG, seed_halu_dialog),
        (HALU_SUMMARIZATION, seed_halu_summarization),
    ],
)


def main(args: Namespace) -> None:
    benchmark = args.benchmark
    kolena.initialize(verbose=True)
    if benchmark:
        proc[benchmark](args.sample_count)
    else:
        # seed all
        for dataset, func in proc.items():
            print(f"seeding {dataset}")
            func(args.sample_count)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--benchmark",
        choices=[SQUAD2_DEV, SQUAD2_TRAIN, HALU_QA, HALU_DIALOG, HALU_SUMMARIZATION],
        help="Name of the benchmark to seed.",
    )
    ap.add_argument("--sample-count", default=0, type=int, help="Number of samples")

    main(ap.parse_args())
