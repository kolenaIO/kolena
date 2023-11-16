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

import kolena
from kolena._experimental.dataset import register_dataset

BUCKET = "kolena-public-datasets"


def seed_squad2_dev():
    dev_json = pd.read_json(f"s3://{BUCKET}/SQuAD2/dev-v2.0.json")

    dev = pd.DataFrame(
        [
            {
                "title": r["title"],
                "context": p["context"],
                "question": qas["question"],
                "id": qas["id"],
                "answers": qas["answers"],
                "is_impossible": qas["is_impossible"],
                "plausible_answers": qas.get("plausible_answers", None),
            }
            for r in dev_json["data"]
            for p in r["paragraphs"]
            for qas in p["qas"]
        ],
    )

    register_dataset("SQuAD 2.0 Dev", dev)


def seed_squad2_train():
    train_json = pd.read_json(f"s3://{BUCKET}/SQuAD2/train-v2.0.json")

    train = pd.DataFrame(
        [
            {
                "title": r["title"],
                "context": p["context"],
                "question": qas["question"],
                "id": qas["id"],
                "answers": qas["answers"],
                "is_impossible": qas["is_impossible"],
                "plausible_answers": qas.get(
                    "plausible_answers",
                    None,
                ),
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

    register_dataset("SQuAD 2.0 Train", train)


def seed_halu_qa():
    qa = pd.read_json(f"{BUCKET}/HaluEval/data/qa_data.json", lines=True)
    register_dataset("HaLuEval qa", qa.rename(columns={"knowledge": "text"}))


def seed_halu_dialog():
    dialogue = pd.read_json(f"{BUCKET}/HaluEval/data/dialogue_data.json", lines=True)
    register_dataset("HaLuEval dialogue", dialogue.rename(columns={"knowledge": "text"}))


def seed_halu_summarization():
    summarization = pd.read_json(f"s3://{BUCKET}/HaluEval/data/summarization_data.json", lines=True)
    register_dataset("HaLuEval summarization", summarization.rename(columns={"document": "text"}))


def main(args: Namespace) -> int:
    benchmark = args.benchmark
    kolena.initialize()
    if benchmark == "squad2-dev":
        seed_squad2_dev()
    elif benchmark == "sqaud2-train":
        seed_squad2_train()
    elif benchmark == "halu-qa":
        seed_halu_qa()
    elif benchmark == "halu-dialog":
        seed_halu_dialog()
    elif benchmark == "halu-summarization":
        seed_halu_summarization()
    else:
        # seed all
        seed_squad2_train()
        seed_squad2_dev()
        seed_halu_qa()
        seed_halu_dialog()
        seed_halu_summarization()

    return 0


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--benchmark",
        choices=["squad2-dev", "squad2-train", "halu-qa", "halu-dialog", "halu-summarization"],
        help="Name of the benchmark to seed.",
    )

    main(ap.parse_args())
