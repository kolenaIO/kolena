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

import pandas as pd

import kolena
from .workflow import GroundTruth
from .workflow import TestCase
from .workflow import TestSample
from .workflow import TestSuite
from kolena.workflow.annotation import ClassificationLabel

kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)


BUCKET = "kolena-public-datasets"
PREFIX = "imagenet1k/validation"
INFERENCES_VIT_BASE_16 = "results/predictions_google_vit_base_16.csv"
INFERENCES_VIT_BASE_32 = "results/predictions_google_vit_base_32.csv"


def locator_as_label(locator: str) -> str:
    label = Path(locator).stem[:-4]
    use_supercategory_as_label = ["other"]
    if label2supercategory[label] in use_supercategory_as_label:
        return label2supercategory[label]
    return label


df_label2supercategory = pd.read_csv(f"s3://{BUCKET}/{PREFIX}/label_to_super_category.csv")
label2supercategory: Dict[str, str] = {}
supercategory2labellist = defaultdict(list)
for row in df_label2supercategory.itertuples():
    label2supercategory[row.label] = row.super_category
    supercategory2labellist[row.super_category].append(row.label)

df_base_16 = pd.read_csv(f"s3://{BUCKET}/{PREFIX}/{INFERENCES_VIT_BASE_16}")
df_base_32 = pd.read_csv(f"s3://{BUCKET}/{PREFIX}/{INFERENCES_VIT_BASE_32}")


test_samples = [
    (
        TestSample(locator=row.locator),
        GroundTruth(classification=ClassificationLabel(label=locator_as_label(row.locator))),
    )
    for row in df_base_16.itertuples()
]
complete_test_case = TestCase("ImageNet-1k :: complete", test_samples=test_samples, reset=True)
labels = {gt.classification.label for _, gt in test_samples}
supercategories = {supercat for supercat in supercategory2labellist}


test_cases = [complete_test_case]
for supercat in supercategories:
    filtered_test_samples = [
        (ts, gt)
        for ts, gt in test_samples
        if (supercat == "other" and gt.classification.label == "other")
        or (gt.classification.label != "other" and label2supercategory[gt.classification.label] == supercat)
    ]
    test_cases.append(
        TestCase(f"ImageNet-1k :: {supercat}", test_samples=filtered_test_samples, reset=True),
    )

test_suite = TestSuite("ImageNet-1k :: 12 supercategories", test_cases=test_cases, reset=True)

# model_to_inferences = {
#     "Google ViT Base16 (ImageNet1k)": df_base_16,
#     "Google ViT Base32 (ImageNet1k)": df_base_32,
# }

# for model_name, df in model_to_inferences.items():
#     locator_to_inference = {}
#     for _, row in df.iterrows():
#         inference_labels = []
#         for label in labels:
#             if label == "other":
#                 confidence = max([row[_label] for _label in supercategory2labellist["other"]])
#             else:
#                 confidence = row[label]

#             inference_labels.append(InferenceLabel(label=label, confidence=confidence))

#         inference_labels.sort(key=lambda x: x.confidence, reverse=True)
#         locator_to_inference[row.locator] = Inference(inferences=inference_labels)

#     def infer(test_sample: TestSample) -> Inference:
#         return locator_to_inference[test_sample.locator]

#     model = Model(model_name, infer=infer)
#     print(f"Seeding results for model: {model_name}")
#     test(model, test_suite, reset=True)
#     print(f"Done seeding results for model: {model_name}")
