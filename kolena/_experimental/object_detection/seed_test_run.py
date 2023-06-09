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
from typing import List
from typing import Tuple

import pandas as pd

import kolena
from kolena._experimental.object_detection.evaluator import ObjectDetectionEvaluator
from kolena._experimental.object_detection.workflow import Inference
from kolena._experimental.object_detection.workflow import Model
from kolena._experimental.object_detection.workflow import TestSample
from kolena._experimental.object_detection.workflow import TestSuite
from kolena._experimental.object_detection.workflow import ThresholdConfiguration
from kolena._experimental.object_detection.workflow import ThresholdStrategy
from kolena.workflow.annotation import ScoredLabeledBoundingBox
from kolena.workflow.test_run import test


DATASET = "coco-2014-val"
WORKFLOW = "OD"
MIN_CONFIDENCE = 0.05
LABELS_OF_INTEREST = {
    "person",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "train",
    "truck",
    "traffic light",
    "fire hydrant",
    "stop sign",
}

MODEL_LIST: Dict[str, str] = {
    "yolo_r": f"YOLOR-D6 (modified CSP, {WORKFLOW})",  # run with sys.argv[1] == 0
    "yolo_x": f"YOLOX (modified CSP-v5, {WORKFLOW})",
    "mask_cnn": f"Mask R-CNN (Inception-ResNet-v2, {WORKFLOW})",
    "faster_rcnn": f"Faster R-CNN (Inception-ResNet-v2, {WORKFLOW})",
    "yolo_v4s": f"Scaled YOLOv4 (CSP-DarkNet-53, {WORKFLOW})",
    "yolo_v3": f"YOLOv3 (DarkNet-53, {WORKFLOW})",
}

TEST_SUITE_NAMES = [
    f"{DATASET} benchmark [Object Detection] :: supercategory",
]

LINK = "s3://kolena-dev-models/object-detection"
GH_LINK = "https://github.com/"

YOLO_R = {
    "model family": "YOLO",
    "published year": "2021",
    "model framework": "PyTorch 1.7.0",
    "training dataset": "coco-2017",
    "image size": "1280x1280",
    "model locator": f"{LINK}/yolor/yolor-d6-paper-573.pt",
    "model source": f"{GH_LINK}WongKinYiu/yolor",
    "tested by": "@jimmysrinivasan",
    "vcs.notebook": f"{GH_LINK}kolenaIO/object-detection-demo/blob/trunk/models/yolor/YOLOR.ipynb",
    "vcs.hash": "eaa1334d4d42cb86ea4f013c58167db4be865eba",
}

YOLO_X = {
    "model family": "YOLO",
    "published year": "2021",
    "model framework": "PyTorch 1.8.0",
    "training dataset": "coco-2017",
    "image size": "640x640",
    "model locator": f"{LINK}/yolox/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth",
    "model source": f"{GH_LINK}open-mmlab/mmdetection/blob/master/configs/yolox/README.md",
    "tested by": "@jeremysato",
    "vcs.notebook": f"{GH_LINK}kolenaIO/object-detection-demo/blob/trunk/models/yolox/YOLOX.ipynb",
    "vcs.hash": "d1c6c8add6004de97038a9c72e8dc430d35a2fb4",
}

MASK_RCNN = {
    "model family": "RCNN",
    "published year": "2017",
    "model framework": "TensorFlow2.5",
    "training dataset": "coco-2017",
    "image size": "1024x1024",
    "model locator": f"{LINK}/mask-rcnn/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz",
    "model source": f"{GH_LINK}tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md",
    "tested by": "@johnshaw",
    "vcs.notebook": f"{GH_LINK}kolenaIO/object-detection-demo/blob/trunk/models/mask-rcnn/mask-rcnn.ipynb",
    "vcs.hash": "1bf28df8ae8cda4e9054a58bbe6840d543f65c6e",
}

FAST_RCNN = {
    "model family": "RCNN",
    "published year": "2016",
    "model framework": "TensorFlow2.5",
    "training dataset": "coco-2017",
    "image size": "1024x1024",
    "model locator": f"{LINK}/faster-rcnn/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.tar.gz",
    "model source": f"{GH_LINK}tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md",
    "tested by": "@janesmith",
    "vcs.notebook": f"{GH_LINK}kolenaIO/object-detection-demo/blob/trunk/models/faster-rcnn/faster-rcnn.ipynb",
    "vcs.hash": "1bf28df8ae8cda4e9054a58bbe6840d543f65c6e",
}

YOLO_V4 = {
    "model family": "YOLO",
    "published year": "2020",
    "model framework": "PyTorch 1.6.0",
    "training dataset": "coco-2017",
    "image size": "1536x1536",
    "model locator": f"{LINK}/yolov4-scaled/yolov4-p7.pt",
    "model source": f"{GH_LINK}WongKinYiu/ScaledYOLOv4",
    "tested by": "@jennystone",
    "vcs.notebook": f"{GH_LINK}kolenaIO/object-detection-demo/blob/trunk/models/yolov4-scaled/scaled-YOLOv4.ipynb",
    "vcs.hash": "eaa1334d4d42cb86ea4f013c58167db4be865eba",
}

YOLO_V3 = {
    "model family": "YOLO",
    "published year": "2018",
    "model framework": "PyTorch 1.5.0",
    "training dataset": "coco-2017",
    "image size": "608x608",
    "model locator": f"{LINK}/yolov3/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth",
    "model source": f"{GH_LINK}open-mmlab/mmdetection/blob/master/configs/yolo/README.md",
    "tested by": "@jillsadeh",
    "vcs.notebook": f"{GH_LINK}kolenaIO/object-detection-demo/blob/trunk/models/yolov3/YOLOv3.ipynb",
    "vcs.hash": "d1c6c8add6004de97038a9c72e8dc430d35a2fb4",
}

MODEL_METADATA: Dict[str, Dict[str, str]] = {
    "yolo_r": YOLO_R,
    "mask_cnn": MASK_RCNN,
    "faster_rcnn": FAST_RCNN,
    "yolo_x": YOLO_X,
    "yolo_v3": YOLO_V3,
    "yolo_v4s": YOLO_V4,
}


def load_inferences(
    min_conf_score: float,
    df_frame_results: pd.DataFrame,
) -> List[ScoredLabeledBoundingBox]:
    return [
        ScoredLabeledBoundingBox(  # type: ignore
            label=str(record.label),
            top_left=(float(record.min_x), float(record.min_y)),  # type: ignore
            bottom_right=(float(record.max_x), float(record.max_y)),  # type: ignore
            score=float(record.confidence_score),
        )
        for record in df_frame_results.itertuples()
        if float(record.confidence_score) > min_conf_score and str(record.label) in LABELS_OF_INTEREST
    ]


def seed_test_run(
    mod: Tuple[str, str],
    min_conf_score: float,
    test_suite: TestSuite,
    groups_df: pd.DataFrame,
) -> None:
    def get_stored_inferences(
        min_conf_score: float,
        meta_df_by_img: pd.DataFrame,
    ) -> Callable[[TestSample], Inference]:
        def infer(sample: TestSample) -> Inference:
            try:
                image_loc: str = sample.locator
                image_name: str = image_loc.split("/")[-1]
                this_group = meta_df_by_img.get_group(r"imgs/" + image_name)
                return Inference(
                    bboxes=load_inferences(min_conf_score, this_group),
                )
            except Exception as e:
                if e is None:
                    print(e)  # ignored
                return Inference(bboxes=[])

        return infer

    model_name = mod[0]
    print(f"working on {model_name} and {test_suite.name} v{test_suite.version}")
    infer = get_stored_inferences(min_conf_score, groups_df)

    model = Model(f"{mod[1]}", infer=infer, metadata=MODEL_METADATA[model_name])

    evaluator_configurations = [
        ThresholdConfiguration(
            threshold_strategy=ThresholdStrategy.FIXED_05,
            iou_threshold=0.5,
            min_confidence_score=min_conf_score,
        ),
        ThresholdConfiguration(
            threshold_strategy=ThresholdStrategy.F1_OPTIMAL,
            iou_threshold=0.5,
            min_confidence_score=min_conf_score,
        ),
        ThresholdConfiguration(
            threshold_strategy=ThresholdStrategy.FIXED_075,
            iou_threshold=0.75,
            min_confidence_score=min_conf_score,
        ),
    ]

    test(model, test_suite, ObjectDetectionEvaluator, evaluator_configurations, reset=True)


def run(args: Namespace) -> None:
    mod = MODEL_LIST[args.model_name]
    df_results = pd.read_csv(
        args.inference_csv,
        dtype={
            "relative_path": object,
            "label": object,
            "confidence_score": object,
            "min_x": object,
            "min_y": object,
            "max_x": object,
            "max_y": object,
        },
    )

    df_results = df_results[df_results.label.isin(LABELS_OF_INTEREST)]
    meta_df_by_img = df_results.groupby("relative_path")
    model_info = (args.model_name, mod)
    if args.test_suite == "none":
        for name in TEST_SUITE_NAMES:
            seed_test_run(model_info, MIN_CONFIDENCE, TestSuite(name), meta_df_by_img)
    else:
        seed_test_run(model_info, MIN_CONFIDENCE, TestSuite(args.test_suite), meta_df_by_img)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--model_name", type=str, help=f"One of {MODEL_LIST.keys()} models.")
    ap.add_argument("--inference_csv", type=str, help="Path to the CSV for the model's inferences.")
    ap.add_argument("--test_suite", type=str, default="none", help="Name of the test suite to run.")
    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)

    run(ap.parse_args())


# poetry run python3
# kolena/_experimental/object_detection/seed_test_run.py
# --model_name "yolo_x"
# --inference_csv "/Users/markchen/Desktop/models/yolo_x/coco-2014-val_prediction.csv"
# --test_suite "coco-2014-val benchmark [Object Detection] :: supercategory"

if __name__ == "__main__":
    main()
