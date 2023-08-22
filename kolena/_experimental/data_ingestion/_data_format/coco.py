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
from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Type

from kolena._experimental.data_ingestion._data_format.base import BaseDataFormat
from kolena._experimental.data_ingestion._types import DataIngestionConfig
from kolena._experimental.data_ingestion._utils.io import load_json
from kolena._experimental.object_detection import GroundTruth
from kolena._experimental.object_detection import Model
from kolena._experimental.object_detection import TestCase
from kolena._experimental.object_detection import TestSample
from kolena._experimental.object_detection import TestSuite
from kolena.workflow import Model as BaseModel
from kolena.workflow import TestCase as BaseTestCase
from kolena.workflow import TestSuite as BaseTestSuite
from kolena.workflow.annotation import LabeledBoundingBox


class CocoDataFormat(BaseDataFormat):
    def __init__(self, config: DataIngestionConfig) -> None:
        super().__init__(config)
        self.raw_data = None

    def load_data(self) -> None:
        self.raw_data = load_json(self.config.data_path)

    def process_data(self) -> None:
        self.image_map: Dict[int, Any] = self._get_image_map(self.raw_data)
        self.label_map: Dict[int, str] = self._get_label_map(self.raw_data)
        self.bbox_map = self._get_bbox_map(self.raw_data, self.image_map, self.label_map)

    def save_data(self) -> None:
        locator_prefix = self.config.locator_prefix
        test_case_name = self.config.test_case_name
        test_suite_name = self.config.test_suite_name
        reset = self.config.reset

        complete_test_case = self._create_complete_test_case(
            self.bbox_map,
            self.image_map,
            reset,
            test_case_name,
            locator_prefix,
        )
        # create the test suite with the complete test case
        _ = TestSuite(
            test_suite_name,
            description="data ingestion via Kolena package",
            test_cases=[complete_test_case],
            reset=reset,
        )

    def list_workflow(self) -> Tuple[Type[BaseTestCase], Type[BaseTestSuite], Type[BaseModel]]:
        return TestCase, TestSuite, Model

    def _get_image_map(self, raw_data):
        return {int(record["id"]): record for record in raw_data["images"]}

    def _get_label_map(self, raw_data):
        return {int(record["id"]): record["name"] for record in raw_data["categories"]}

    def _get_bbox_map(self, raw_data, image_map, label_map):
        ids = set(image_map.keys())
        image_to_boxes: Dict[int, List[LabeledBoundingBox]] = defaultdict(lambda: [])
        for annotation in raw_data["annotations"]:
            image_id = annotation["image_id"]
            label = label_map[annotation["category_id"]]

            # check that the box is in a valid image, not a crowd box
            if image_id in ids and not annotation["iscrowd"]:
                bbox = annotation["bbox"]
                top_left = (bbox[0], bbox[1])
                bottom_right = (bbox[0] + bbox[2], bbox[1] + bbox[3])
                bounding_box = LabeledBoundingBox(top_left, bottom_right, label)
                image_to_boxes[image_id].append(bounding_box)
        return image_to_boxes

    def _create_complete_test_case(
        self,
        bbox_map,
        image_map,
        reset: bool,
        test_case_name: str,
        locator_prefix: str,
    ) -> TestCase:
        # create a test sample object and a ground truth object per image
        test_samples_and_ground_truths: List[Tuple[TestSample, GroundTruth]] = []
        for image_id, image_record in image_map.items():
            image_name = image_record["file_name"]
            test_sample = TestSample(
                locator=locator_prefix + image_name,
                metadata={
                    "image_width": image_record["width"],
                    "image_height": image_record["height"],
                },
            )

            ground_truth = GroundTruth(
                bboxes=bbox_map[image_id],
                ignored_bboxes=[],
            )

            test_samples_and_ground_truths.append((test_sample, ground_truth))

        # create the complete test case, not attached to any test suite
        complete_test_case = TestCase(
            test_case_name,
            test_samples=test_samples_and_ground_truths,
            reset=reset,
        )

        return complete_test_case
