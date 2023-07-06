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
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import pandas as pd

import kolena
from kolena._experimental.classification.workflow import GroundTruth
from kolena._experimental.classification.workflow import ImageTestCase
from kolena._experimental.classification.workflow import ImageTestSample
from kolena._experimental.classification.workflow import ImageTestSuite
from kolena._experimental.classification.workflow import TextTestCase
from kolena._experimental.classification.workflow import TextTestSample
from kolena._experimental.classification.workflow import TextTestSuite
from kolena._experimental.classification.workflow import VideoTestCase
from kolena._experimental.classification.workflow import VideoTestSample
from kolena._experimental.classification.workflow import VideoTestSuite
from kolena._experimental.classification.workflow import WORKFLOW_TYPES
from kolena.workflow.annotation import ClassificationLabel
from kolena.workflow.test_sample import Metadata


def TestSample(
    locator_or_text: str,
    metadata: Metadata,
    workflow_type: WORKFLOW_TYPES = WORKFLOW_TYPES.TEXT,
) -> Union[ImageTestSample, VideoTestSample, TextTestSample]:
    if workflow_type == WORKFLOW_TYPES.IMAGE:
        return ImageTestSample(locator=locator_or_text, metadata=metadata)
    elif workflow_type == WORKFLOW_TYPES.VIDEO:
        return VideoTestSample(locator=locator_or_text, metadata=metadata)
    elif workflow_type == WORKFLOW_TYPES.TEXT:
        return TextTestSample(text=locator_or_text, metadata=metadata)


def TestCase(
    name: str,
    version: Optional[int] = None,
    description: Optional[str] = None,
    test_samples: Optional[List[Tuple[Union[ImageTestSample, VideoTestSample, TextTestSample], GroundTruth]]] = None,
    reset: bool = False,
    workflow_type: WORKFLOW_TYPES = WORKFLOW_TYPES.TEXT,
) -> Union[ImageTestCase, VideoTestCase, TextTestCase]:
    if workflow_type == WORKFLOW_TYPES.IMAGE:
        WorkflowTestCase = ImageTestCase
    elif workflow_type == WORKFLOW_TYPES.VIDEO:
        WorkflowTestCase = VideoTestCase
    elif workflow_type == WORKFLOW_TYPES.TEXT:
        WorkflowTestCase = TextTestCase

    return WorkflowTestCase(
        name=name,
        version=version,
        description=description,
        test_samples=test_samples,
        reset=reset,
    )


def TestSuite(
    name: str,
    version: Optional[int] = None,
    description: Optional[str] = None,
    test_cases: Optional[List[Union[ImageTestCase, VideoTestCase, TextTestCase]]] = None,
    reset: bool = False,
    tags: Optional[Set[str]] = None,
    workflow_type: WORKFLOW_TYPES = WORKFLOW_TYPES.TEXT,
) -> Union[ImageTestSuite, VideoTestSuite, TextTestSuite]:
    if workflow_type == WORKFLOW_TYPES.IMAGE:
        WorkflowTestSuite = ImageTestSuite
    elif workflow_type == WORKFLOW_TYPES.VIDEO:
        WorkflowTestSuite = VideoTestSuite
    elif workflow_type == WORKFLOW_TYPES.TEXT:
        WorkflowTestSuite = TextTestSuite

    return WorkflowTestSuite(
        name=name,
        version=version,
        description=description,
        test_cases=test_cases,
        reset=reset,
        tags=tags,
    )


# seed_test_suite.py


kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)


BUCKET = "kolena-public-datasets"
PREFIX = "imagenet1k/validation"
INFERENCES_VIT_BASE_16 = "results/predictions_google_vit_base_16.csv"
INFERENCES_VIT_BASE_32 = "results/predictions_google_vit_base_32.csv"
WORKFLOW_TYPE = WORKFLOW_TYPES.IMAGE


def locator_as_label(locator: str) -> str:
    label = Path(locator).stem[:-4]
    use_supercategory_as_label = ["other"]
    if label2supercategory[label] in use_supercategory_as_label:
        return label2supercategory[label]
    return label


df_label2supercategory = pd.read_csv("/Users/markchen/Desktop/Imagenet-1k/label_to_super_category.csv")
label2supercategory: Dict[str, str] = {}
supercategory2labellist = defaultdict(list)
for row in df_label2supercategory.itertuples():
    label2supercategory[row.label] = row.super_category
    supercategory2labellist[row.super_category].append(row.label)

df_base_16 = pd.read_csv("/Users/markchen/Desktop/Imagenet-1k/predictions_google_vit_base_16.csv")
df_base_32 = pd.read_csv("/Users/markchen/Desktop/Imagenet-1k/predictions_google_vit_base_16.csv")


test_samples = [
    (
        TestSample(locator_or_text=row.locator, metadata={}, workflow_type=WORKFLOW_TYPE),
        GroundTruth(classification=ClassificationLabel(label=locator_as_label(row.locator))),
    )
    for row in df_base_16.itertuples()
]
complete_test_case = TestCase(
    "ImageNet-1k :: complete [Prebuilt Multiclass Classification]",
    test_samples=test_samples,
    reset=True,
    workflow_type=WORKFLOW_TYPE,
)
test_suite = TestSuite(
    "ImageNet-1k :: 12 supercategories [Prebuilt Multiclass Classification]",
    test_cases=[complete_test_case],
    reset=True,
    workflow_type=WORKFLOW_TYPE,
)
