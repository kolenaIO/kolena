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
from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import pandas as pd

from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import ScoredLabeledBoundingBox


def load_data(
    df_metadata_csv: pd.DataFrame,
    is_pred: bool = False,
) -> pd.DataFrame:
    image_to_boxes: Dict[str, List[Union[ScoredLabeledBoundingBox, LabeledBoundingBox, BoundingBox]]] = defaultdict(
        list,
    )
    image_to_metadata: Dict[str, Dict[str, Any]] = defaultdict(dict)

    for record in df_metadata_csv.itertuples():
        coords = (float(record.min_x), float(record.min_y)), (float(record.max_x), float(record.max_y))
        bounding_box = (
            LabeledBoundingBox(*coords, record.label)
            if (not is_pred)
            else ScoredLabeledBoundingBox(*coords, record.label, record.confidence_score)
        )
        image_to_boxes[record.locator].append(bounding_box)

        if not is_pred:
            metadata = {
                "height": float(record.height),
                "width": float(record.width),
                "date_captured": str(record.date_captured),
                "supercategory": str(record.supercategory),
                "brightness": float(record.brightness),
            }
            image_to_metadata[record.locator] = metadata

    df_boxes = pd.DataFrame(
        list(image_to_boxes.items()),
        columns=["locator", "bounding_boxes" if not is_pred else "raw_inferences"],
    )
    df_metadata = pd.DataFrame(list(image_to_metadata.items()), columns=["locator", "metadata"])

    return df_boxes.merge(df_metadata, on="locator") if not is_pred else df_boxes
