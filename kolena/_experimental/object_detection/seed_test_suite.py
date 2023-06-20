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
from argparse import ArgumentParser
from argparse import Namespace
from collections import defaultdict
from typing import Any
from typing import Callable
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

DATASET = "coco-2014-val-car-and-cats"
S3_PATH = "s3://kolena-public-datasets/coco-2014-val/imgs/"
PERSON_LABELS = {"person"}
ANIMAL_LABELS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
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
    "bench",
    "boat",
    "airplane",
    "parking meter",
}
VALID_ID = 24  # ignore animal, person and transportation IDs are lower
SUITE_DESCRIPTION = f"All images in the {DATASET} dataset"


def create_complete_test_case(args: Namespace) -> TestCase:
    test_samples_and_ground_truths: List[Tuple[TestSample, GroundTruth]] = []
    id_to_bboxes: Dict[int, List[LabeledBoundingBox]] = defaultdict(lambda: [])
    coco_data, ids, label_map = load_coco_data(args.annotations)

    # preprocess annotations
    for annotation in coco_data["annotations"]:
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if 1 < category_id <= VALID_ID and not annotation["iscrowd"]:
            bbox = annotation["bbox"]
            top_left = (bbox[0], bbox[1])
            bottom_right = (bbox[0] + bbox[2], bbox[1] + bbox[3])
            label = label_map[category_id]
            bounding_box = LabeledBoundingBox(top_left, bottom_right, label)
            id_to_bboxes[image_id].append(bounding_box)

    for image_id in tqdm(ids):
        image_name = "COCO_val2014_" + str(image_id).zfill(12) + ".jpg"

        # calculating blur and brightness is slow
        im = Image.open(args.images + image_name).convert("L")  # to grayscale
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
                    bboxes=bounding_boxes,
                    ignored_bboxes=[],
                ),
            ),
        )

    complete_test_case = TestCase(
        f"complete :: {DATASET} [Object Detection]",
        description=SUITE_DESCRIPTION,
        test_samples=test_samples_and_ground_truths,
        reset=True,
    )

    return complete_test_case


def seed_test_suite_by_supercategory(test_suite_name: str, complete_test_case: TestCase) -> None:
    test_case_name_to_decision_logic_map = {
        # "person": lambda cat: cat in PERSON_LABELS,
        "animal": lambda cat: cat in ANIMAL_LABELS,
        "transportation": lambda cat: cat in TRANSPORTATION_LABELS,
    }

    # test_case_name_to_decision_logic_map = {
    #     "person": lambda cat: cat not in TRANSPORTATION_LABELS and cat not in ANIMAL_LABELS,
    #     "animal": lambda cat: cat not in TRANSPORTATION_LABELS and cat not in PERSON_LABELS,
    #     "transportation": lambda cat: cat not in ANIMAL_LABELS and cat not in PERSON_LABELS,
    # }

    test_cases = []
    for name, fn in test_case_name_to_decision_logic_map.items():
        ts_list = []
        for ts, gt in complete_test_case.iter_test_samples():
            if not gt.bboxes or any([fn(box.label) for box in gt.bboxes]):
                gtbboxes = []
                ignoredbboxes = []

                for box in gt.bboxes:
                    if fn(box.label):
                        gtbboxes.append(box)
                    else:
                        ignoredbboxes.append(box)

                ts_list.append(
                    (
                        ts,
                        GroundTruth(
                            bboxes=gtbboxes,
                            ignored_bboxes=ignoredbboxes + gt.ignored_bboxes,
                        ),
                    ),
                )

        new_ts = TestCase(
            f"supercategory :: {name} :: {DATASET}",
            description=f"Any images with {name} objects in the {DATASET} dataset.",
            test_samples=ts_list,
            reset=True,
        )
        test_cases.append(new_ts)

    test_suite = TestSuite(
        test_suite_name,
        description=f"{SUITE_DESCRIPTION}, stratified by the `person`, `animal`, and `transportation` supercategories",
        test_cases=[complete_test_case, *test_cases],
        reset=True,
    )
    print(f"created test suite {test_suite.name} v{test_suite.version}")


def seed_test_suites(
    test_suite_names: Dict[str, Callable[[str, TestCase], TestSuite]],
    complete_test_case: TestCase,
) -> None:
    for test_suite_name, test_suite_fn in test_suite_names.items():
        test_suite_fn(test_suite_name, complete_test_case)


def load_coco_data(file_path) -> Tuple[Any, List[int], Dict[int, str]]:
    with open(file_path) as file:
        coco_data = json.load(file)
        ids = sorted({int(entry["id"]) for entry in coco_data["images"]})
        label_map: Dict[int, str] = {int(category["id"]): category["name"] for category in coco_data["categories"]}
        return coco_data, ids, label_map


def run(args: Namespace) -> None:
    print("creating complete test case")
    complete_test_case = create_complete_test_case(args)
    test_suite_names: Dict[str, Callable[[str, TestCase], TestSuite]] = {
        f"{DATASET} benchmark [Object Detection] :: supercategory": seed_test_suite_by_supercategory,
    }

    print("creating test suites")
    seed_test_suites(test_suite_names, complete_test_case)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--annotations", help="Local path to the JSON annotation file")
    ap.add_argument("--images", help="Local path to images of the dataset")
    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
