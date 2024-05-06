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
import pickle
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path

import pandas as pd
import s3fs
from crossing_pedestrian_detection.constants import BUCKET
from crossing_pedestrian_detection.constants import DATASET
from crossing_pedestrian_detection.constants import DEFAULT_DATASET_NAME
from crossing_pedestrian_detection.constants import ID_FIELDS
from crossing_pedestrian_detection.utils import process_gt_bboxes
from smart_open import open as smart_open
from tqdm import tqdm

from kolena.dataset import upload_dataset


def video_locator(video_path: str) -> str:
    return f"s3://{video_path}"


def thumbnail_locator(filename: str) -> str:
    return f"s3://{BUCKET}/{DATASET}/data/images/{filename}/00000.png"


def process_data() -> pd.DataFrame:
    s3 = s3fs.S3FileSystem(anon=True)
    video_files = s3.glob(f"{BUCKET}/{DATASET}/data/videos/*.mp4")
    raw_data_pkl = f"s3://{BUCKET}/{DATASET}/raw/jaad_database.pkl"
    with smart_open(raw_data_pkl, "rb") as gt_file:
        gt_annotations = pickle.load(gt_file)

    datapoints = []
    for video_file in tqdm(video_files):
        filename = Path(video_file).stem
        processed_gts = process_gt_bboxes(gt_annotations[filename]["ped_annotations"])

        if len(processed_gts.high_risk_bboxes) > 0:
            datapoints.append(
                {
                    "locator": video_locator(video_file),
                    "video_id": int(filename.split("_")[-1]),
                    "filename": filename,
                    "thumbnail_locator": thumbnail_locator(filename),
                    "num_frames": gt_annotations[filename]["num_frames"],
                    "width": gt_annotations[filename]["width"],
                    "height": gt_annotations[filename]["height"],
                    "time_of_day": gt_annotations[filename]["time_of_day"],
                    "weather": gt_annotations[filename]["weather"],
                    "location": gt_annotations[filename]["location"],
                    "high_risk": processed_gts.high_risk_bboxes,
                    "low_risk": processed_gts.low_risk_bboxes,
                    "high_risk_pids": processed_gts.high_risk_pids,
                    "low_risk_pids": processed_gts.low_risk_pids,
                    "n_pedestrians": len(list(gt_annotations[filename]["ped_annotations"].keys())),
                },
            )

    return pd.DataFrame(datapoints)


def run(args: Namespace) -> int:
    df = process_data()
    upload_dataset(args.dataset, df, id_fields=ID_FIELDS)
    return 0


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Optionally specify a custom dataset name to upload.",
    )
    run(ap.parse_args())
