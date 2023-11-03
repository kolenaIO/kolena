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
from concurrent.futures import ThreadPoolExecutor
from typing import List

from semantic_segmentation.utils import create_bitmap
from semantic_segmentation.utils import download_binary_array
from semantic_segmentation.utils import upload_image_buffer

from kolena._utils.log import info
from kolena._utils.log import progress_bar


class ActivationMapUploader:
    def __init__(self, inf_locator_prefix: str, map_locator_prefix: str):
        self.pool = ThreadPoolExecutor()
        self.inf_locator_prefix = inf_locator_prefix
        self.map_locator_prefix = map_locator_prefix

    def submit(self, test_sample_names: List[str]) -> None:
        info("generating and uploading activation maps...")
        for _ in progress_bar(
            self.pool.map(self.process_activation_map, test_sample_names),
            total=len(test_sample_names),
        ):
            pass
        info("finished uploading activation maps")

    def process_activation_map(self, test_sample_name: str) -> None:
        prob_array = download_binary_array(self.inf_locator_prefix + f"{test_sample_name}_person.npy")
        activation_map = create_bitmap(prob_array)
        upload_image_buffer(self.map_locator_prefix + f"{test_sample_name}.png", activation_map)
