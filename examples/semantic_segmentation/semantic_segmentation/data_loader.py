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
import functools
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from typing import List
from typing import Tuple

import numpy as np
from pydantic.dataclasses import dataclass
from semantic_segmentation.utils import create_bitmap
from semantic_segmentation.utils import download_binary_array
from semantic_segmentation.utils import upload_image
from semantic_segmentation.utils import upload_image_buffer
from semantic_segmentation.workflow import GroundTruth
from semantic_segmentation.workflow import Inference
from semantic_segmentation.workflow import Label
from semantic_segmentation.workflow import TestSample

from kolena.workflow.annotation import SegmentationMask


@dataclass(frozen=True)
class ResultMask:
    type: str  # "TP", "FP", "FN"
    mask: SegmentationMask
    count: int


ResultMasks = Tuple[ResultMask, ResultMask, ResultMask]
ResultMasksByLocator = Tuple[str, ResultMask, ResultMask, ResultMask]


class DataLoader:
    def __init__(self):
        self.pool = ThreadPoolExecutor()

    def load_batch(
        self,
        batch: List[Tuple[TestSample, GroundTruth, Inference]],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        def load(ts: TestSample, gt: GroundTruth, inf: Inference) -> Tuple[str, np.ndarray, np.ndarray]:
            inf_prob = download_binary_array(inf.prob.locator)
            gt_mask = np.zeros_like(inf_prob)
            return ts.locator, gt_mask, inf_prob

        futures = [self.pool.submit(functools.partial(load, *item)) for item in batch]
        successes, failures = wait(futures)
        if len(failures) != 0:
            exceptions = ", ".join([str(failure.exception()) for failure in failures])
            raise RuntimeError(f"failed to load {len(failures)} samples: {exceptions}")

        # splice together correct ordering
        gt_by_locator = {result[0]: result[1] for result in [f.result() for f in successes]}
        inf_by_locator = {result[0]: result[2] for result in [f.result() for f in successes]}
        return [gt_by_locator[ts.locator] for ts, _, _ in batch], [inf_by_locator[ts.locator] for ts, _, _ in batch]

    def upload_batch(
        self,
        locator_prefix: str,
        batch: List[Tuple[TestSample, np.ndarray, np.ndarray]],
    ) -> List[ResultMasks]:
        def upload(ts: TestSample, gt_mask: np.ndarray, inf_mask: np.ndarray) -> ResultMasksByLocator:
            def upload_result_mask(category: str, mask: np.ndarray) -> ResultMask:
                locator = f"{locator_prefix}/{category}/{ts.metadata['basename']}.png"
                upload_image(locator, mask)
                return ResultMask(
                    type=category,
                    mask=SegmentationMask(locator=locator, labels=Label.as_label_map()),
                    count=np.sum(mask),
                )

            tp = upload_result_mask("TP", np.where(gt_mask != inf_mask, 0, inf_mask))
            fp = upload_result_mask("FP", np.where(gt_mask == inf_mask, 0, inf_mask))
            fn = upload_result_mask("FN", np.where(gt_mask == inf_mask, 0, gt_mask))
            return ts.locator, tp, fp, fn

        futures = [self.pool.submit(functools.partial(upload, *item)) for item in batch]
        successes, failures = wait(futures)
        if len(failures) != 0:
            exceptions = ", ".join([str(failure.exception()) for failure in failures])
            raise RuntimeError(f"failed to upload {len(failures)} samples: {exceptions}")

        # splice together correct ordering
        result_masks_by_locator = {result[0]: tuple(result[1:]) for result in [f.result() for f in successes]}
        return [result_masks_by_locator[ts.locator] for ts, _, _ in batch]


class ActivationMapUploader:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1)
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
            raise RuntimeError(f"missing uploads for {len(self.futures) - len(successes)} activation maps")
        else:
            print(f"completed upload of {len(self.futures)} activation maps")

    def submit(self, prob_array_locator: str, activation_map_locator: str) -> None:
        if activation_map_locator in self.futures.keys():
            # already being processed, skip
            return
        future = self.pool.submit(
            functools.partial(self.process_activation_map, prob_array_locator, activation_map_locator),
        )
        self.futures[activation_map_locator] = future

    def process_activation_map(self, prob_array_locator: str, activation_map_locator: str) -> None:
        prob_array = download_binary_array(prob_array_locator)
        activation_map = create_bitmap(prob_array)
        upload_image_buffer(activation_map_locator, activation_map)
