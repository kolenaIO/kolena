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
from examples.question_answering.question_answering.utils import normalize_string
from kolena._experimental.dataset import register_dataset
from kolena.workflow.annotation import Utterance

BUCKET = "kolena-public-datasets"
DATASET = "CoQA"


def main(args: Namespace) -> None:
    kolena.initialize(verbose=True)

    df = pd.read_csv(args.dataset_csv)
    context_dict = {
        x[0]: x[1]
        for x in df[["data_id", "question", "metadata_answer"]]
        .groupby("data_id")
        .agg(list)
        .reset_index()
        .apply(
            lambda x: [
                x["data_id"],
                [
                    (Utterance(text=q, group="question"), Utterance(text=a, group="answer"))
                    for q, a in zip(x["question"], x["metadata_answer"])
                ],
            ],
            axis=1,
            result_type="expand",
        )
        .to_dict("records")
    }
    df["context"] = df.apply(lambda x: [e for y in context_dict[x["data_id"]][: x["turn"]] for e in y], axis=1)
    df["text"] = df.apply(
        lambda x: x["story"]
        + "\n\n"
        + "\n".join(
            [f"{t.group[0].upper()}: {t.text}" for e in context_dict[x["data_id"]][: x["turn"]] for t in e],
        ),
        axis=1,
    )
    df["clean_answer"] = df["metadata_answer"].apply(normalize_string)

    register_dataset(
        DATASET,
        df.rename(
            columns={
                "metadata_answer": "answer",
                "metadata_wc_answer": "wc_answer",
                "metadata_wc_number": "wc_number",
            },
        ),
    )


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset_csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/metadata/metadata.csv",
        help="CSV file with a stories, questions, and answers.",
    )
    main(ap.parse_args())
