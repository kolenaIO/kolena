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
from video_retrieval_by_text.constants import DATASET_URI
from video_retrieval_by_text.constants import DEFAULT_DATASET_NAME
from video_retrieval_by_text.constants import ID_FIELDS
from video_retrieval_by_text.constants import S3_STORAGE_OPTIONS

from kolena.asset import ImageAsset
from kolena.asset import VideoAsset
from kolena.dataset.dataset import upload_dataset


def prep_data(df: pd.DataFrame) -> pd.DataFrame:
    df["video"] = df.apply(
        lambda row: VideoAsset(locator=row.locator, thumbnail=ImageAsset(row.thumbnail), start=row.start, end=row.end),
        axis=1,
    )
    df["video_res"] = df.apply(lambda row: f"{int(row.video_width)}x{int(row.video_height)}", axis=1)
    return df[["caption_id", "caption", "caption_writing_level", "word_count", "video", "video_res"]]


def generate_video_asset(row: pd.Series) -> pd.Series:
    row.video = VideoAsset(locator=row.locator, thumbnail=ImageAsset(row.thumbnail), start=row.start, end=row.end)
    return row


def run(args: Namespace) -> None:
    vatex_df = pd.read_csv(DATASET_URI, storage_options=S3_STORAGE_OPTIONS)
    df = prep_data(vatex_df)
    upload_dataset(args.name, df, id_fields=ID_FIELDS)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument(
        "-n",
        "--name",
        default=DEFAULT_DATASET_NAME,
        type=str,
        help="Optionally specify a custom name for the dataset.",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
