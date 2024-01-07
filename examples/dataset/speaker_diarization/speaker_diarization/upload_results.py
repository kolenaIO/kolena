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

import pandas as pd
from speaker_diarization.data_loader import DATASET
from speaker_diarization.data_loader import load_data
from speaker_diarization.data_loader import load_results
from speaker_diarization.utils import compute_metrics
from speaker_diarization.utils import realign_labels

import kolena
from kolena.dataset import upload_results


def align_df_speakers(ref: pd.Series, inf: pd.Series) -> pd.Series:
    return pd.Series([realign_labels(r, i) for r, i in zip(ref.tolist(), inf.tolist())])


def run(args: Namespace) -> None:
    kolena.initialize(verbose=True)

    model = "gcp-stt-video"
    align_speakers = args.align_speakers
    sample_count = args.sample_count

    datapoint_df = load_data()
    transcripts_df = load_results(model)
    datapoint_df.sort_values(by="transcription_path", inplace=True, ignore_index=True)
    transcripts_df.sort_values(by="transcription_path", inplace=True, ignore_index=True)

    if sample_count:
        datapoint_df = datapoint_df[:sample_count]
        transcripts_df = transcripts_df[:sample_count]

    if align_speakers:
        transcripts_df["transcripts"] = align_df_speakers(datapoint_df["transcripts"], transcripts_df["transcripts"])

    metrics_df = compute_metrics(datapoint_df, transcripts_df)
    merged_df = pd.concat([transcripts_df, metrics_df], axis=1)
    results_df = datapoint_df[["locator", "transcription_path"]].merge(
        merged_df,
        on="transcription_path",
        validate="one_to_one",
    )
    eval_config = {"align-speakers": align_speakers}
    upload_results(args.dataset, model, [(eval_config, results_df)])


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--dataset", type=str, default=DATASET, help="Optionally specify a dataset name to upload.")
    ap.add_argument(
        "--align-speakers",
        action="store_true",
        help="Specify whether to perform speaker alignment between the ground_truth and inference in the "
        "preprocessing step.",
    )
    ap.add_argument(
        "--sample-count",
        type=int,
        default=0,
        help="Number of samples to use. All samples are used by default.",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
