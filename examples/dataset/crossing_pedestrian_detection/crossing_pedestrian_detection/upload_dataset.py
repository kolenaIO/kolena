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
from crossing_pedestrian_detection.utils import process_gt_bboxes
from smart_open import open as smart_open
from tqdm import tqdm

import kolena
from kolena.annotation import ClassificationLabel
from kolena.annotation import LabeledBoundingBox

from kolena.dataset import upload_dataset


def video_locator(video_path: str) -> str:
    return f"s3://{video_path}"


def thumbnail_locator(filename: str) -> str:
    return f"s3://{BUCKET}/{DATASET}/images/{filename}/00000.png"



def process_data() -> pd.DataFrame:
    s3 = s3fs.S3FileSystem(anon=True)
    video_files = s3.glob(f"{BUCKET}/{DATASET}/JAAD_clips/*.mp4")
    raw_data_pkl = f"s3://{BUCKET}/{DATASET}/data_cache/jaad_database.pkl"
    with smart_open(raw_data_pkl, "rb") as gt_file:
        gt_annotations = pickle.load(gt_file)

    datapoints = []
    for video_file in tqdm(video_files):
        filename = Path(video_file).stem
        bboxes, risk_pids = process_gt_bboxes(gt_annotations[filename]["ped_annotations"])

        if len(bboxes) > 0:
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
                    "high_risk": bboxes[0],
                    "low_risk": bboxes[1],
                    "high_risk_pids": risk_pids[0],
                    "low_risk_pids": risk_pids[1],
                    "n_pedestrians": len(list(gt_annotations[filename]["ped_annotations"].keys())),
                },
            )

    return pd.DataFrame(datapoints)


if __name__ == "__main__":
    kolena.initialize(verbose=True)
    df = process_data()
    upload_dataset("JAAD [crossing-pedestrian-detection]", df, id_fields=ID_FIELDS)
