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
from face_recognition_11.constants import DATASET
from face_recognition_11.constants import DATASET_DETECTION
from face_recognition_11.constants import DATASET_METADATA
from face_recognition_11.constants import DATASET_PAIRS
from face_recognition_11.constants import TASK

import kolena
from kolena.annotation import BoundingBox
from kolena.annotation import Keypoints
from kolena.asset import ImageAsset
from kolena.dataset import upload_dataset


def run(args: Namespace) -> None:
    kolena.initialize(verbose=True)

    df_metadata = pd.read_csv(DATASET_METADATA)
    df_pairs = pd.read_csv(DATASET_PAIRS)
    df_detection = pd.read_csv(DATASET_DETECTION)

    genuine_pairs = {}
    imposter_pairs = {}
    for record in df_pairs.itertuples():
        if record.is_match:
            genuine_pairs[record.locator_1] = record.locator_2
        else:
            imposter_pairs[record.locator_1] = record.locator_2

    df_raw_data = df_detection.merge(df_metadata, on="locator", how="left")

    datapoints = []
    for record in df_raw_data.itertuples():
        datapoints.append(
            dict(
                locator=record.locator,
                pairs=[
                    ImageAsset(locator=genuine_pairs[record.locator], is_match=True),  # type: ignore
                    ImageAsset(locator=imposter_pairs[record.locator], is_match=False),  # type: ignore
                ],
                bbox=BoundingBox(
                    top_left=(record.min_x, record.min_y),
                    bottom_right=(record.max_x, record.max_y),
                ),
                keypoints=Keypoints(
                    points=[
                        (record.left_eye_x, record.left_eye_y),
                        (record.right_eye_x, record.right_eye_y),
                    ],
                ),
                normalization_factor=record.normalization_factor,
                person=record.person,
                age=record.age,
                race=record.race,
                gender=record.gender,
                width=record.width,
                height=record.height,
            ),
        )

    df_datapoints = pd.DataFrame(datapoints)
    upload_dataset(args.dataset, df_datapoints)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        default=f"{DATASET} [{TASK}]",
        help="Optionally specify a custom dataset name to upload.",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
