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


def seed_squad2_dev():
    register_dataset("SQuAD 2.0 Dev", load_squad2_dev())


def seed_squad2_train():
    register_dataset("SQuAD 2.0 Train", load_squad2_train())


def seed_halu_qa():
    register_dataset("HaLuEval qa", load_halu_qa())


def seed_halu_dialog():
    register_dataset("HaLuEval dialogue", load_halu_dialog())


def seed_halu_summarization():
    register_dataset("HaLuEval summarization", load_halu_summarization())


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
        proc[benchmark]()
    else:
        # seed all
        for dataset, func in proc.items():
            print(f"seeding {dataset}")
            func()


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--benchmark",
        choices=[SQUAD2_DEV, SQUAD2_TRAIN, HALU_QA, HALU_DIALOG, HALU_SUMMARIZATION],
        help="Name of the benchmark to seed.",
    )

    main(ap.parse_args())
