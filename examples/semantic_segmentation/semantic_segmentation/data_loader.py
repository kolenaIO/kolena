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
from semantic_segmentation.utils import download_binary_array
from semantic_segmentation.utils import download_mask
from semantic_segmentation.utils import upload_image
from semantic_segmentation.workflow import GroundTruth
from semantic_segmentation.workflow import Inference
from semantic_segmentation.workflow import Label
from semantic_segmentation.workflow import TestSample

from kolena.workflow.annotation import SegmentationMask

ResultMasks = Tuple[SegmentationMask, SegmentationMask, SegmentationMask]
ResultMasksByLocator = Tuple[str, SegmentationMask, SegmentationMask, SegmentationMask]


class DataLoader:
    def __init__(self):
        self.pool = ThreadPoolExecutor()

    def load_batch(
        self,
        batch: List[Tuple[TestSample, GroundTruth, Inference]],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        def load(ts: TestSample, gt: GroundTruth, inf: Inference) -> Tuple[str, np.ndarray, np.ndarray]:
            inf_prob = download_binary_array(inf.prob.locator)
            gt_mask = download_mask(gt.mask.locator)
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
            def upload_result_mask(category: str, mask: np.ndarray) -> SegmentationMask:
                locator = f"{locator_prefix}/{category}/{ts.metadata['basename']}.png"
                upload_image(locator, mask)
                return SegmentationMask(locator=locator, labels=Label.as_label_map())

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
