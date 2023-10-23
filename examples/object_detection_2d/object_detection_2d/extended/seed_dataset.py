# Copyright 2021-2023 Kolena Inc.
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
import json
from argparse import ArgumentParser
from argparse import Namespace
from collections import defaultdict
from typing import Dict
from typing import List

import pandas as pd
import s3fs
from object_detection_2d.constants import DATASET
from object_detection_2d.constants import S3_ANNOTATION_FILE_PATH
from object_detection_2d.constants import S3_IMAGE_LOCATION
from object_detection_2d.constants import TRANSPORTATION_LABELS

import kolena
from kolena._experimental.dataset._dataset import register_dataset
from kolena.workflow.annotation import LabeledBoundingBox


def load_transportation_data() -> Dict[str, List[LabeledBoundingBox]]:
    s3 = s3fs.S3FileSystem()

    try:
        with s3.open(S3_ANNOTATION_FILE_PATH, "r") as file:
            coco_data = json.load(file)
    except OSError as e:
        print(e, "\nPlease ensure you have set up AWS credentials.")
        exit()

    # gather image IDs with the Attribution 2.0 license - https://creativecommons.org/licenses/by/2.0/
    ids = {int(entry["id"]) for entry in coco_data["images"] if entry["license"] == 4}

    # class id to string label
    label_map: Dict[int, str] = {int(category["id"]): category["name"] for category in coco_data["categories"]}

    # cache bounding boxes per image
    image_to_boxes: Dict[str, List[LabeledBoundingBox]] = defaultdict(lambda: [])
    for annotation in coco_data["annotations"]:
        image_id = annotation["image_id"]
        label = label_map[annotation["category_id"]]

        # check that the box is in a valid image, not a crowd box, and in the transportation supercategory
        if image_id in ids and not annotation["iscrowd"] and label in TRANSPORTATION_LABELS:
            bbox = annotation["bbox"]
            top_left = (bbox[0], bbox[1])
            bottom_right = (bbox[0] + bbox[2], bbox[1] + bbox[3])
            bounding_box = LabeledBoundingBox(top_left, bottom_right, label)
            image_to_boxes["COCO_val2014_" + str(image_id).zfill(12) + ".jpg"].append(bounding_box)
    return image_to_boxes


def complete_datapoints(args: Namespace) -> pd.DataFrame:
    print("loading S3 files...")
    image_to_boxes = load_transportation_data()
    data = pd.read_csv(args.metadata)
    dataset = data[["width", "height", "brightness"]]
    dataset["locator"] = S3_IMAGE_LOCATION + data["name"]
    dataset["bboxes"] = data["name"].apply(lambda x: image_to_boxes[x])
    dataset["ignored_bboxes"] = data["name"].apply(lambda x: [])

    return dataset


def main(args: Namespace) -> None:
    kolena.initialize(verbose=True)

    register_dataset(args.dataset, complete_datapoints(args))


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--dataset", type=str, default=DATASET, help="Dataset name")
    ap.add_argument(
        "--metadata",
        type=str,
        default="s3://kolena-public-datasets/coco-2014-val/meta/metadata_attribution2.0_brightness.csv",
        help="The path to the image metadata stored in a csv.",
    )

    main(ap.parse_args())
