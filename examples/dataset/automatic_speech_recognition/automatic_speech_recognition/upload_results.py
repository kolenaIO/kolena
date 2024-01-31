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
from argparse import ArgumentParser
from argparse import Namespace
from typing import Any
from typing import Dict

import langcodes
import langid
import pandas as pd
from automatic_speech_recognition.constants import BUCKET
from automatic_speech_recognition.constants import DATASET
from automatic_speech_recognition.constants import MODEL_NAME
from automatic_speech_recognition.utils import generate_diff_word_level
from automatic_speech_recognition.utils import preprocess_transcription
from jiwer import cer
from jiwer import process_words
from tqdm import tqdm

import kolena
from kolena.dataset import download_dataset
from kolena.dataset import upload_results


def compute_metrics(ground_truth: str, inference: str) -> Dict[str, Any]:
    diff = generate_diff_word_level(ground_truth, inference)
    wer_metrics = process_words(ground_truth, inference)
    word_errors = diff["ins_count"] + diff["sub_count"] + diff["del_count"]
    word_error_rate = wer_metrics.wer
    match_error_rate = wer_metrics.mer
    word_information_lost = wer_metrics.wil
    word_information_preserved = wer_metrics.wip
    character_error_rate = cer(ground_truth, inference)
    language = langcodes.Language.get(langid.classify(inference)[0]).display_name()

    return dict(
        is_failure=word_errors != 0,
        word_errors=word_errors,
        word_error_rate=word_error_rate,
        match_error_rate=match_error_rate,
        word_information_lost=word_information_lost,
        word_information_preserved=word_information_preserved,
        character_error_rate=character_error_rate,
        insertion_count=diff["ins_count"],
        deletion_count=diff["del_count"],
        substitution_count=diff["sub_count"],
        language=language,
    )


def run(args: Namespace) -> None:
    kolena.initialize(verbose=True)
    df_dataset = download_dataset(args.dataset)
    df = pd.read_csv(f"s3://{BUCKET}/{DATASET}/results/raw/{args.model}.csv", storage_options={"anon": True})
    df["transcript"] = df["transcript"].str.lower().apply(preprocess_transcription)
    df = df.merge(df_dataset[["id", "transcript"]], on="id", suffixes=("_inf", "_gt"))

    results = []
    for record in tqdm(df.itertuples(), total=len(df)):
        metrics = compute_metrics(record.transcript_gt, record.transcript_inf)
        results.append(
            dict(
                id=record.id,
                transcript=record.transcript_inf,
                **metrics,
            ),
        )

    df_results = pd.DataFrame.from_records(results)
    upload_results(args.dataset, MODEL_NAME[args.model], df_results)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument(
        "model",
        type=str,
        default="whisper-default",
        nargs="?",
        choices=list(MODEL_NAME.keys()),
        help="Name of the model to test.",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default=DATASET,
        help="Optionally specify a custom dataset name to test.",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
