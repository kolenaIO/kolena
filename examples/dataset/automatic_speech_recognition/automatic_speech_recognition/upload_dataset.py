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
from automatic_speech_recognition.constants import BUCKET
from automatic_speech_recognition.constants import DATASET
from automatic_speech_recognition.utils import preprocess_transcription

from kolena.asset import AudioAsset
from kolena.dataset import upload_dataset


def run(args: Namespace) -> None:
    df_dataset = pd.read_csv(args.dataset_csv, storage_options={"anon": True})
    df_dataset["audio"] = df_dataset["audio"].apply(AudioAsset)
    df_dataset["transcript"] = df_dataset["transcript"].str.lower().apply(preprocess_transcription)
    upload_dataset(args.dataset_name, df_dataset, id_fields=["id"])


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset-csv",
        type=str,
        default=f"s3://{BUCKET}/{DATASET}/raw/LibriSpeech.csv",
        help="CSV file specifying dataset. See default CSV for details",
    )
    ap.add_argument(
        "--dataset-name",
        type=str,
        default=DATASET,
        help="Optionally specify a name of the dataset",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
