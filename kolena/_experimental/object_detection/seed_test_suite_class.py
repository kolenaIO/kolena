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
import os
from argparse import Namespace
from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import kolena._experimental
from kolena._experimental.object_detection.workflow import GroundTruth
from kolena._experimental.object_detection.workflow import TestCase
from kolena._experimental.object_detection.workflow import TestSample
from kolena._experimental.object_detection.workflow import TestSuite
from kolena.workflow.annotation import LabeledBoundingBox

DATASET = "coco-2014-val"
S3_PATH = "s3://kolena-public-datasets/coco-2014-val/imgs/"
PERSON_LABELS = {"person"}
TRANSPORTATION_LABELS = {
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "train",
    "truck",
    "traffic light",
    "fire hydrant",
    "stop sign",
}
VALID_ID = 14  # last animal, person and transportation IDs are lower
SUITE_DESCRIPTION = f"All images in the {DATASET} dataset"

# poetry run python3 kolena/_experimental/object_detection/seed_test_suite_class.py
# --annotations "/Users/markchen/Desktop/Kolena/coco-2014-val/instances_val2014.json"
# --images "/Users/markchen/Desktop/val2014/"
annotations = "/Users/markchen/Desktop/Kolena/coco-2014-val/instances_val2014.json"
images = "/Users/markchen/Desktop/val2014/"
id_to_bboxes: Dict[int, List[LabeledBoundingBox]] = defaultdict(lambda: [])


def load_coco_data(file_path) -> Tuple[Any, List[int], Dict[int, str]]:
    with open(file_path) as file:
        coco_data = json.load(file)
        ids = sorted({int(entry["id"]) for entry in coco_data["images"]})
        label_map: Dict[int, str] = {int(category["id"]): category["name"] for category in coco_data["categories"]}
        return coco_data, ids, label_map


coco_data, ids, label_map = load_coco_data(annotations)
# preprocess annotations
for annotation in coco_data["annotations"]:
    image_id = int(annotation["image_id"])
    category_id = int(annotation["category_id"])
    if category_id <= VALID_ID and not annotation["iscrowd"]:
        bbox = annotation["bbox"]
        top_left = (bbox[0], bbox[1])
        bottom_right = (bbox[0] + bbox[2], bbox[1] + bbox[3])
        label = label_map[category_id]
        bounding_box = LabeledBoundingBox(top_left, bottom_right, label)
        id_to_bboxes[image_id].append(bounding_box)


def create_complete_case() -> TestCase:
    test_samples_and_ground_truths: List[Tuple[TestSample, GroundTruth]] = []

    filename = "/Users/markchen/Desktop/kolena-2/kolena/_experimental/object_detection/class_person.txt"

    with open(filename) as file:
        lines = file.readlines()
        lines = [int(line.strip()) for line in lines]

    ids = lines

    filename = "/Users/markchen/Desktop/kolena-2/kolena/_experimental/object_detection/class_transportation.txt"

    with open(filename) as file:
        lines = file.readlines()
        lines = [int(line.strip()) for line in lines]

    ids = lines + ids
    ids = list(set(ids))

    for image_id in tqdm(ids):
        image_name = "COCO_val2014_" + str(image_id).zfill(12) + ".jpg"

        # calculating blur and brightness is slow
        im = Image.open(images + image_name).convert("L")  # to grayscale
        width, height = im.size
        array = np.asarray(im, dtype=np.int32)
        gy, gx = np.gradient(array)
        blur, brightness = np.average(np.sqrt(gx**2 + gy**2)), np.average(array)
        bounding_boxes = id_to_bboxes[image_id]

        test_samples_and_ground_truths.append(
            (
                TestSample(
                    locator=S3_PATH + image_name,
                    metadata={
                        "image_width": width,
                        "image_height": height,
                        "blur": blur,
                        "brightness": brightness,
                    },
                ),
                GroundTruth(
                    bboxes=[bb for bb in bounding_boxes if bb.label in PERSON_LABELS],
                    ignored_bboxes=[bb for bb in bounding_boxes if bb.label in TRANSPORTATION_LABELS],
                ),
            ),
        )

    complete_test_case = TestCase(
        f"person-copy :: {DATASET} [Object Detection]",
        description=SUITE_DESCRIPTION,
        test_samples=test_samples_and_ground_truths,
        reset=True,
    )

    return complete_test_case


def create_complete_person_case(args: Namespace) -> TestCase:
    test_samples_and_ground_truths: List[Tuple[TestSample, GroundTruth]] = []

    filename = "/Users/markchen/Desktop/kolena-2/kolena/_experimental/object_detection/class_person.txt"

    with open(filename) as file:
        lines = file.readlines()
        lines = [int(line.strip()) for line in lines]

    ids = lines

    for image_id in tqdm(ids):
        image_name = "COCO_val2014_" + str(image_id).zfill(12) + ".jpg"

        # calculating blur and brightness is slow
        im = Image.open(images + image_name).convert("L")  # to grayscale
        width, height = im.size
        array = np.asarray(im, dtype=np.int32)
        gy, gx = np.gradient(array)
        blur, brightness = np.average(np.sqrt(gx**2 + gy**2)), np.average(array)
        bounding_boxes = id_to_bboxes[image_id]

        test_samples_and_ground_truths.append(
            (
                TestSample(
                    locator=S3_PATH + image_name,
                    metadata={
                        "image_width": width,
                        "image_height": height,
                        "blur": blur,
                        "brightness": brightness,
                    },
                ),
                GroundTruth(
                    bboxes=[bb for bb in bounding_boxes if bb.label in PERSON_LABELS],
                    ignored_bboxes=[bb for bb in bounding_boxes if bb.label in TRANSPORTATION_LABELS],
                ),
            ),
        )

    complete_test_case = TestCase(
        f"person-copy :: {DATASET} [Object Detection]",
        description=SUITE_DESCRIPTION,
        test_samples=test_samples_and_ground_truths,
        reset=True,
    )

    return complete_test_case


def create_complete_transpo_case(args: Namespace) -> TestCase:
    test_samples_and_ground_truths: List[Tuple[TestSample, GroundTruth]] = []

    filename = "/Users/markchen/Desktop/kolena-2/kolena/_experimental/object_detection/class_transportation.txt"

    with open(filename) as file:
        lines = file.readlines()
        lines = [int(line.strip()) for line in lines]

    ids = lines

    for image_id in tqdm(ids):
        image_name = "COCO_val2014_" + str(image_id).zfill(12) + ".jpg"

        # calculating blur and brightness is slow
        im = Image.open(images + image_name).convert("L")  # to grayscale
        width, height = im.size
        array = np.asarray(im, dtype=np.int32)
        gy, gx = np.gradient(array)
        blur, brightness = np.average(np.sqrt(gx**2 + gy**2)), np.average(array)
        bounding_boxes = id_to_bboxes[image_id]

        test_samples_and_ground_truths.append(
            (
                TestSample(
                    locator=S3_PATH + image_name,
                    metadata={
                        "image_width": width,
                        "image_height": height,
                        "blur": blur,
                        "brightness": brightness,
                    },
                ),
                GroundTruth(
                    bboxes=[bb for bb in bounding_boxes if bb.label in TRANSPORTATION_LABELS],
                    ignored_bboxes=[bb for bb in bounding_boxes if bb.label in PERSON_LABELS],
                ),
            ),
        )

    complete_test_case = TestCase(
        f"transportation-copy :: {DATASET} [Object Detection]",
        description=SUITE_DESCRIPTION,
        test_samples=test_samples_and_ground_truths,
        reset=True,
    )

    return complete_test_case


kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)

base = create_complete_case()
x = create_complete_person_case()
y = create_complete_transpo_case()

test_suite = TestSuite(
    f"{DATASET} benchmark [Object Detection] :: supercategory-copy",
    description=f"{SUITE_DESCRIPTION}, stratified by the `person`, and `transportation` supercategories",
    test_cases=[base, x, y],
    reset=True,
)
