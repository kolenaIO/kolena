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

from kolena.annotation import BoundingBox
from kolena.annotation import Keypoints
from kolena.asset import ImageAsset
from kolena.dataset import upload_dataset


def run(args: Namespace) -> None:
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
    df_raw_data["image_asset"] = df_raw_data.apply(
        lambda x: ImageAsset(
            locator=x.locator,
            width=x.width,  # type: ignore[call-arg]
            height=x.height,  # type: ignore[call-arg]
            normalization_factor=x.normalization_factor,  # type: ignore[call-arg]
            bbox=BoundingBox(
                top_left=(x.min_x, x.min_y),
                bottom_right=(
                    x.max_x,
                    x.max_y,
                ),
            ),  # type: ignore[call-arg]
            keypoints=Keypoints(
                points=[
                    (x.left_eye_x, x.left_eye_y),
                    (x.right_eye_x, x.right_eye_y),
                ],
            ),  # type: ignore[call-arg]
        ),
        axis=1,
    )
    df_raw_data["person"] = df_raw_data.apply(
        lambda x: dict(
            person=x.person,
            age=x.age,
            race=x.race,
            gender=x.gender,
        ),
        axis=1,
    )
    df_raw_data = df_raw_data[["locator", "image_asset", "person"]]
    df_pairs = df_pairs.merge(df_raw_data, left_on="locator_1", right_on="locator")
    df_pairs = df_pairs.merge(df_raw_data, left_on="locator_2", right_on="locator")
    df_pairs["pairs"] = df_pairs.apply(
        lambda x: [
            ImageAsset(**x.image_asset_x.__dict__, position="left"),  # type: ignore[call-arg]
            ImageAsset(**x.image_asset_y.__dict__, position="right"),  # type: ignore[call-arg]
        ],
        axis=1,
    )
    df_pairs = df_pairs.rename(columns={"person_x": "left", "person_y": "right"})
    df_pairs = df_pairs[["locator_1", "locator_2", "left", "right", "pairs", "is_match"]]

    upload_dataset(args.dataset, df_pairs, id_fields=["locator_1", "locator_2"])


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
