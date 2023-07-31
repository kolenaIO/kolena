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
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import pandas as pd
import s3fs
from object_detection_2d.constants import DATASET
from object_detection_2d.constants import S3_ANNOTATION_FILE_PATH
from object_detection_2d.constants import S3_IMAGE_LOCATION
from object_detection_2d.constants import TEST_SUITE_DESCRIPTION
from object_detection_2d.constants import TRANSPORTATION_LABELS
from object_detection_2d.constants import WORKFLOW

import kolena
from kolena._experimental.object_detection import GroundTruth
from kolena._experimental.object_detection import TestCase
from kolena._experimental.object_detection import TestSample
from kolena._experimental.object_detection import TestSuite
from kolena.workflow.annotation import BoundingBox
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
    image_to_boxes: Dict[int, List[LabeledBoundingBox]] = defaultdict(lambda: [])
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


def create_complete_transportation_case(args: Namespace) -> TestCase:
    print("loading S3 files...")
    image_to_boxes = load_transportation_data()
    data = pd.read_csv(args.metadata)

    # create a test sample object and a ground truth object per image
    test_samples_and_ground_truths: List[Tuple[TestSample, GroundTruth]] = []
    for _, row in data.iterrows():
        image_name = row["name"]
        test_sample = TestSample(
            locator=S3_IMAGE_LOCATION + image_name,
            metadata={
                "image_width": row["width"],
                "image_height": row["height"],
                "brightness": row["brightness"],
            },
        )

        ground_truth = GroundTruth(
            bboxes=image_to_boxes[image_name],
            ignored_bboxes=[],
        )

        test_samples_and_ground_truths.append((test_sample, ground_truth))

    # create the complete test case, not attached to any test suite
    complete_test_case = TestCase(
        f"complete transportation :: {DATASET}",
        test_samples=test_samples_and_ground_truths,
        reset=True,
    )

    return complete_test_case


def seed_test_suite_by_brightness(test_suite_name: str, complete_test_case: TestCase) -> None:
    stratification_logic_map = {
        "light": lambda brightness: brightness >= 130,
        "normal": lambda brightness: 100 <= brightness < 130,
        "dark": lambda brightness: 0 <= brightness < 100,
    }

    # create each test case by stratification
    test_cases: List[TestCase] = []
    for name, fn in stratification_logic_map.items():
        filtered_test_samples = [
            (ts, gt) for ts, gt in complete_test_case.iter_test_samples() if fn(ts.metadata["brightness"])
        ]
        new_test_case = TestCase(
            f"brightness :: {name} :: {DATASET}",
            test_samples=filtered_test_samples,
            reset=True,
        )
        test_cases.append(new_test_case)

    # create the test suite with the complete test case and new test cases
    test_suite = TestSuite(
        test_suite_name,
        description=f"{TEST_SUITE_DESCRIPTION}, stratified by `light`, `normal`, and `dark` brightness",
        test_cases=[complete_test_case, *test_cases],
        reset=True,
    )
    print(f"created test suite '{test_suite.name}' v{test_suite.version}")


def seed_test_suite_by_bounding_box_size(test_suite_name: str, complete_test_case: TestCase) -> None:
    stratification_logic_map = {
        "small": lambda area: area < 10_000,
        "medium": lambda area: 10_000 <= area < 60_000,
        "large": lambda area: 60_000 <= area,
    }

    def filter_gt_bboxes(gt: GroundTruth, filter_fn: Callable[[float], bool]) -> GroundTruth:
        bboxes, ignored_bboxes = [], gt.ignored_bboxes
        for bbox in gt.bboxes:
            bboxes.append(bbox) if filter_fn(_area(bbox)) else ignored_bboxes.append(bbox)
        return GroundTruth(bboxes=bboxes, ignored_bboxes=ignored_bboxes)

    # create each test case by stratification
    test_cases: List[TestCase] = []
    for name, fn in stratification_logic_map.items():
        samples_with_filtered_bboxes = []
        for ts, gt in complete_test_case.iter_test_samples():
            filtered_ground_truth = filter_gt_bboxes(gt, fn)
            if filtered_ground_truth.n_bboxes > 0:
                samples_with_filtered_bboxes.append((ts, filtered_ground_truth))
        new_test_case = TestCase(
            f"bounding box size :: {name} :: {DATASET}",
            test_samples=samples_with_filtered_bboxes,
            reset=True,
        )
        test_cases.append(new_test_case)

    # create the test suite with the complete test case and new test cases
    test_suite = TestSuite(
        test_suite_name,
        description=f"{TEST_SUITE_DESCRIPTION}, stratified by `small`, `medium`, and `large` sized bounding boxes",
        test_cases=[complete_test_case, *test_cases],
        reset=True,
    )
    print(f"created test suite '{test_suite.name}' v{test_suite.version}")


def _area(bbox: BoundingBox) -> float:
    width = bbox.bottom_right[0] - bbox.top_left[0]
    height = bbox.bottom_right[1] - bbox.top_left[1]
    return width * height


def main(args: Namespace) -> None:
    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)

    complete_test_case = create_complete_transportation_case(args)

    # organize test suite names with its generator
    test_suites: List[Tuple[str, Callable[[str, TestCase], TestSuite]]] = [
        (f"{DATASET} :: transportation by brightness [{WORKFLOW}]", seed_test_suite_by_brightness),
        (f"{DATASET} :: transportation by bounding box size [{WORKFLOW}]", seed_test_suite_by_bounding_box_size),
    ]

    # create each test suite using the complete test case
    for test_suite_name, test_suite_creator in test_suites:
        test_suite_creator(test_suite_name, complete_test_case)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--metadata",
        type=str,
        default="s3://kolena-public-datasets/coco-2014-val/meta/metadata_attribution2.0_brightness.csv",
        help="The path to the image metadata stored in a csv.",
    )

    main(ap.parse_args())
