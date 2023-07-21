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
from kolena._experimental.classification import evaluate_classification
from kolena._experimental.classification import GroundTruth
from kolena._experimental.classification import Inference
from kolena._experimental.classification import Model
from kolena._experimental.classification import TestCase
from kolena._experimental.classification import TestSample
from kolena._experimental.classification import TestSuite
from kolena._experimental.classification import ThresholdConfiguration
from kolena.workflow.annotation import ClassificationLabel
from kolena.workflow.annotation import ScoredClassificationLabel
from kolena.workflow.test_run import test

kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)


BUCKET = "kolena-public-datasets"
PREFIX = "imagenet1k/validation"
INFERENCES_VIT_BASE_16 = "results/predictions_google_vit_base_16.csv"
INFERENCES_VIT_BASE_32 = "results/predictions_google_vit_base_32.csv"

# toggle for just the complete test case or stratification by super category
SUPERCAT = False


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
        TestSample(text=row.locator, metadata={}),
        GroundTruth(classification=ClassificationLabel(label=locator_as_label(row.locator))),
    )
    for row in df_base_16.itertuples()
]


# TODO: shrink and filter on classes - make sure seeding by super cat
# maybe not whole workflow, but examples
# binary classification evaluator is not implemented
# test_samples = test_samples[:500]

complete_test_case = TestCase(
    "ImageNet-1k :: complete [Prebuilt Multiclass Classification]",
    test_samples=test_samples,
    reset=True,
)
labels = {gt.classification.label for _, gt in test_samples}
supercategories = {supercat for supercat in supercategory2labellist}


test_cases = [complete_test_case]

if SUPERCAT:
    for supercat in supercategories:
        filtered_test_samples = [
            (ts, gt)
            for ts, gt in test_samples
            if (supercat == "other" and gt.classification.label == "other")
            or (gt.classification.label != "other" and label2supercategory[gt.classification.label] == supercat)
        ]
        if len(filtered_test_samples) > 0:
            test_cases.append(
                TestCase(
                    f"ImageNet-1k :: {supercat} [Prebuilt Multiclass Classification]",
                    test_samples=filtered_test_samples,
                    reset=True,
                ),
            )

test_suite = TestSuite(
    "ImageNet-1k :: 12 supercategories [Prebuilt Multiclass Classification]",
    test_cases=test_cases,
    reset=True,
)


model_to_inferences = {
    "Google ViT Base16 (ImageNet1k, Prebuilt)": df_base_16,
    "Google ViT Base32 (ImageNet1k, Prebuilt)": df_base_32,
}

for model_name, df in model_to_inferences.items():
    locator_to_inference = {}
    for _, row in df.iterrows():
        inference_labels = []
        for label in labels:
            if label == "other":
                confidence = max([row[_label] for _label in supercategory2labellist["other"]])
            else:
                confidence = row[label]

            inference_labels.append(ScoredClassificationLabel(label=label, score=confidence))

        inference_labels.sort(key=lambda x: x.score, reverse=True)
        locator_to_inference[row.locator] = Inference(inference_labels)


def infer(test_sample: TestSample) -> Inference:
    return locator_to_inference[test_sample.locator]


model = Model(model_name, infer=infer)
print(f"Seeding results for model: {model_name}")

configurations = [
    ThresholdConfiguration(
        threshold=0.05,
    ),
    ThresholdConfiguration(
        threshold=None,
    ),
]

test(model, test_suite, evaluator=evaluate_classification, configurations=configurations, reset=True)
