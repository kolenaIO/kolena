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
from pathlib import Path

import pandas as pd
import s3fs
from crossing_pedestrian_detection.constants import BUCKET
from crossing_pedestrian_detection.constants import DATASET
from crossing_pedestrian_detection.constants import ID_FIELDS
from crossing_pedestrian_detection.utils import process_ped_annotations
from smart_open import open as smart_open
from tqdm import tqdm

import kolena
from kolena.annotation import ClassificationLabel
from kolena.dataset import upload_dataset


def video_locator(video_path: str) -> str:
    return f"s3://{video_path}"


def thumbnail_locator(filename: str) -> str:
    return f"s3://{BUCKET}/{DATASET}/images/{filename}/00000.png"


def process_data() -> pd.DataFrame:
    s3 = s3fs.S3FileSystem(anon=True)
    video_files = s3.glob("kolena-public-datasets/JAAD/JAAD_clips/*.mp4")
    raw_data_pkl = f"s3://{BUCKET}/{DATASET}/data_cache/jaad_database.pkl"
    with smart_open(raw_data_pkl, "rb") as gt_file:
        gt_annotations = pickle.load(gt_file)

    datapoints = []
    for video_file in tqdm(video_files):
        filename = Path(video_file).stem
        bboxes_per_ped = process_ped_annotations(gt_annotations[filename]["ped_annotations"])
        bboxes = []

        for pid, ped_bboxes in bboxes_per_ped.items():
            bboxes.extend(ped_bboxes)

        if len(bboxes) > 0:
            datapoints.append(
                {
                    "locator": video_locator(video_file),
                    "ped_id_label": ClassificationLabel(label=pid),
                    "pid": pid,
                    "video_id": int(filename.split("_")[-1]),
                    "filename": filename,
                    "thumbnail_locator": thumbnail_locator(filename),
                    "num_frames": gt_annotations[filename]["num_frames"],
                    "width": gt_annotations[filename]["width"],
                    "height": gt_annotations[filename]["height"],
                    "time_of_day": gt_annotations[filename]["time_of_day"],
                    "weather": gt_annotations[filename]["weather"],
                    "location": gt_annotations[filename]["location"],
                    "focus": bboxes,
                    "is_crossing": bboxes[0].label == "is_crossing",
                    "n_pedestrians": len(bboxes_per_ped),
                },
            )

    return pd.DataFrame(datapoints)


if __name__ == "__main__":
    kolena.initialize(verbose=True)
    df = process_data()
    upload_dataset("JAAD", df, id_fields=ID_FIELDS)
