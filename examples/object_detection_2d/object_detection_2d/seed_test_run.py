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
from argparse import ArgumentParser
from argparse import Namespace
from typing import Callable
from typing import Dict

import pandas as pd
from constants import DATASET
from constants import MODEL_METADATA
from constants import S3_MODEL_INFERENCE_PREFIX
from constants import TRANSPORTATION_LABELS
from constants import WORKFLOW

import kolena
from kolena._experimental.object_detection import Inference
from kolena._experimental.object_detection import Model
from kolena._experimental.object_detection import ObjectDetectionEvaluator
from kolena._experimental.object_detection import TestSample
from kolena._experimental.object_detection import TestSuite
from kolena._experimental.object_detection import ThresholdConfiguration
from kolena._experimental.object_detection import ThresholdStrategy
from kolena.workflow.annotation import ScoredLabeledBoundingBox
from kolena.workflow.test_run import test


MODEL_LIST: Dict[str, str] = {
    "yolo_r": f"YOLOR-D6 (modified CSP, {WORKFLOW})",
    "yolo_x": f"YOLOX (modified CSP-v5, {WORKFLOW})",
    "mask_cnn": f"Mask R-CNN (Inception-ResNet-v2, {WORKFLOW})",
    "faster_rcnn": f"Faster R-CNN (Inception-ResNet-v2, {WORKFLOW})",
    "yolo_v4s": f"Scaled YOLOv4 (CSP-DarkNet-53, {WORKFLOW})",
    "yolo_v3": f"YOLOv3 (DarkNet-53, {WORKFLOW})",
}

# list of all test suites to run if a test suite is not specified
TEST_SUITE_NAMES = [
    f"{DATASET} :: transportation brightness [Object Detection]",
]


def model_alias_to_data_path(alias: str) -> str:
    return S3_MODEL_INFERENCE_PREFIX + alias + "/coco-2014-val_prediction_attribution_2.0_transportation.csv"


# transforms test samples into inferences using a dataframe
def get_stored_inferences(
    metadata_by_image: pd.DataFrame,
) -> Callable[[TestSample], Inference]:
    # a function that creates an inference from a test sample
    def infer(sample: TestSample) -> Inference:
        try:
            image_inferences = metadata_by_image.get_group(sample.locator)
            return Inference(
                bboxes=[
                    ScoredLabeledBoundingBox(
                        top_left=(float(record.min_x), float(record.min_y)),
                        bottom_right=(float(record.max_x), float(record.max_y)),
                        label=str(record.label),
                        score=float(record.confidence_score),
                    )
                    for record in image_inferences.itertuples()
                ],
            )
        except Exception as e:
            if e is None:
                print(e)  # ignored
            return Inference(bboxes=[])

    return infer


def seed_test_run(
    model_alias: str,
    model_full_name: str,
    test_suite: TestSuite,
    groups_df: pd.DataFrame,
) -> None:
    # create a model
    model = Model(model_full_name, infer=get_stored_inferences(groups_df), metadata=MODEL_METADATA[model_alias])

    # customizable configurations for the evaluator
    evaluator = ObjectDetectionEvaluator(
        configurations=[
            ThresholdConfiguration(
                threshold_strategy=ThresholdStrategy.FIXED_03,
                iou_threshold=0.3,
                with_class_level_metrics=True,
                min_confidence_score=0.2,
            ),
            ThresholdConfiguration(
                threshold_strategy=ThresholdStrategy.F1_OPTIMAL,
                iou_threshold=0.5,
                with_class_level_metrics=True,
                min_confidence_score=0.0,
            ),
        ],
    )

    # runs the evaluation
    test(model, test_suite, evaluator, reset=True)


def main(args: Namespace) -> None:
    print("loading csv of inferences from S3...")

    try:
        df_results = pd.read_csv(
            model_alias_to_data_path(args.model),
            dtype={
                "locator": object,
                "label": object,
                "confidence_score": object,
                "min_x": object,
                "min_y": object,
                "max_x": object,
                "max_y": object,
            },
        )
    except OSError as e:
        print(e, "\nPlease ensure you have set up AWS credentials.")
        exit()

    # filter for transportation inferences
    df_results = df_results[df_results.label.isin(TRANSPORTATION_LABELS)]

    # group image inferences together
    metadata_by_image = df_results.groupby("locator")

    model_alias = args.model
    model_full_name = MODEL_LIST[model_alias]

    # run evaluation on test suites
    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)

    if args.test_suite == "none":
        for name in TEST_SUITE_NAMES:
            seed_test_run(model_alias, model_full_name, TestSuite(name), metadata_by_image)
    else:
        seed_test_run(model_alias, model_full_name, TestSuite(args.test_suite), metadata_by_image)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("model", choices=MODEL_LIST.keys(), help="The alias of the model to test.")
    ap.add_argument("--test_suite", type=str, default="none", help="Name of the test suite to run.")

    main(ap.parse_args())
