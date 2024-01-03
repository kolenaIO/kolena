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
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator
from typing import List
from typing import Tuple

import numpy as np
from pydantic.dataclasses import dataclass
from semantic_segmentation.utils import download_binary_array
from semantic_segmentation.utils import download_mask
from semantic_segmentation.utils import upload_image
from semantic_segmentation.workflow import GroundTruth
from semantic_segmentation.workflow import Inference
from semantic_segmentation.workflow import Label
from semantic_segmentation.workflow import TestSample

from kolena._utils.log import progress_bar
from kolena.workflow.annotation import SegmentationMask


@dataclass(frozen=True)
class ResultMask:
    type: str  # "TP", "FP", "FN"
    mask: SegmentationMask
    count: int


ResultMasks = Tuple[ResultMask, ResultMask, ResultMask]


class DataLoader:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=32)

    def download_masks(
        self,
        ground_truths: List[GroundTruth],
        inferences: List[Inference],
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        def load(gt: GroundTruth, inf: Inference) -> Tuple[np.ndarray, np.ndarray]:
            inf_prob = download_binary_array(inf.prob.locator)
            gt_mask = download_mask(gt.mask.locator)
            gt_mask[gt_mask != 1] = 0  # binarize gt_mask
            return gt_mask, inf_prob

        return progress_bar(self.pool.map(load, ground_truths, inferences), total=len(ground_truths))

    def upload_masks(
        self,
        locator_prefix: str,
        test_samples: List[TestSample],
        gt_masks: List[np.ndarray],
        inf_masks: List[np.ndarray],
    ) -> List[ResultMasks]:
        def upload(ts: TestSample, gt_mask: np.ndarray, inf_mask: np.ndarray) -> ResultMasks:
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
            return tp, fp, fn

        return list(
            progress_bar(
                self.pool.map(upload, test_samples, gt_masks, inf_masks),
                total=len(test_samples),
            ),
        )
