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
import pandas as pd

from kolena.annotation import LabeledTimeSegment

BUCKET = "kolena-public-examples"
DATASET = "ICSI-corpus"
DIR = f"s3://{BUCKET}/{DATASET}"
DATA_CSV = f"{DIR}/raw/metadata.csv"
TRANSCRIPTS_CSV = f"{DIR}/raw/transcripts.csv"
DATA_DIR = f"{DIR}/data"
RESULTS_RAW_DIR = f"{DIR}/results/raw"

S3_STORAGE_OPTIONS = {"anon": True}


def _annotate(df: pd.DataFrame) -> list[LabeledTimeSegment]:
    return [
        LabeledTimeSegment(start=r.starttime, end=r.endtime, label=r.label, group=r.speaker) for r in df.itertuples()
    ]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, storage_options=S3_STORAGE_OPTIONS)
    df = df.where(pd.notnull(df), None)  # read missing cells as None
    df["locator"] = df["audio_path"].apply(lambda x: f"{DATA_DIR}/{x}")

    transcript_df = pd.read_csv(TRANSCRIPTS_CSV, storage_options=S3_STORAGE_OPTIONS)

    anno_df = transcript_df.groupby("transcription_path").apply(_annotate).reset_index()
    anno_df.rename(columns={0: "transcripts"}, inplace=True)

    return df.merge(anno_df, on="transcription_path")


def load_results(model: str) -> pd.DataFrame:
    transcript_df = pd.read_csv(f"{RESULTS_RAW_DIR}/{model}.csv", storage_options=S3_STORAGE_OPTIONS)
    results_df = transcript_df.groupby("transcription_path").apply(_annotate).reset_index()
    results_df.rename(columns={0: "transcripts"}, inplace=True)

    return results_df
