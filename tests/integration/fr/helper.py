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
from typing import List

import numpy as np
import pandas as pd

from kolena.fr import ImageResultDataFrame


def generate_image_results(image_ids: List[int]) -> ImageResultDataFrame:
    # unimportant that the chips be different, use a single chip for memory efficiency
    image_chip = (np.random.rand(112, 112, 3) * 256).astype(np.uint8)
    batch_size = len(image_ids)
    df = pd.DataFrame(
        dict(
            image_id=image_ids,
            bounding_box=[np.random.rand(4).astype(np.float32) for _ in range(batch_size)],
            landmarks_input_image=[image_chip] * batch_size,
            landmarks=[np.random.rand(10).astype(np.float32) for _ in range(batch_size)],
            quality_input_image=[image_chip] * batch_size,
            quality=np.random.rand(batch_size).astype(np.float64).tolist(),
            acceptability=np.random.rand(batch_size).astype(np.float64).tolist(),
            fr_input_image=[image_chip] * batch_size,
            embedding=[np.random.rand(256).astype(np.float32) for _ in range(batch_size)],
            failure_reason=[None] * batch_size,
        ),
    )
    return ImageResultDataFrame(df)
