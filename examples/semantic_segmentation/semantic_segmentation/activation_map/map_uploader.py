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
from concurrent.futures import wait

from semantic_segmentation.utils import download_binary_array
from semantic_segmentation.utils import upload_image_buffer
from kolena.workflow.visualization import colorize_activation_map
from kolena.workflow.visualization import encode_png


class ActivationMapUploader:
    def __init__(self):
        self.pool = ThreadPoolExecutor()
        self.futures = {}

    def wait(self) -> None:
        if len(self.futures) == 0:
            return

        print(f"waiting for uploads for {len(self.futures)} activation maps")
        successes, failures = wait(list(self.futures.values()))

        if len(failures) != 0:
            exceptions = ", ".join([str(failure.exception()) for failure in failures])
            raise RuntimeError(f"failed to load {len(failures)} samples: {exceptions}")
        elif len(successes) != len(self.futures):
            raise RuntimeError(
                f"missing uploads for {len(self.futures) - len(successes)} activation maps"
            )
        else:
            print(f"completed upload of {len(self.futures)} activation maps")

    def submit(self, prob_array_locator: str, activation_map_locator: str) -> None:
        if activation_map_locator in self.futures.keys():
            # already being processed, skip
            return
        future = self.pool.submit(
            self.process_activation_map, prob_array_locator, activation_map_locator
        )
        self.futures[activation_map_locator] = future

    def process_activation_map(
        self, prob_array_locator: str, activation_map_locator: str
    ) -> None:
        prob_array = download_binary_array(prob_array_locator)
        bitmap = colorize_activation_map(prob_array)
        activation_map = encode_png(bitmap, mode="RGBA")
        upload_image_buffer(activation_map_locator, activation_map)
