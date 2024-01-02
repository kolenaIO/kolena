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
from typing import Dict

DATASET = "coco-2014-val"
WORKFLOW = "Object Detection"


S3_PREFIX = "s3://"
S3_BUCKET_COCO = "kolena-public-datasets/coco-2014-val"
S3_ANNOTATION_FILE_PATH = f"{S3_BUCKET_COCO}/meta/instances_val2014_attribution_2.0_transportation.json"
S3_IMAGE_LOCATION = f"{S3_PREFIX}{S3_BUCKET_COCO}/imgs/"
S3_MODEL_INFERENCE_PREFIX = f"{S3_PREFIX}{S3_BUCKET_COCO}/results/object_detection/coco_models/"


TEST_SUITE_DESCRIPTION = f"Transportation images from the {DATASET} dataset"
TRANSPORTATION_LABELS = {
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
    "yolo_x": YOLO_X,
    "mask_rcnn": MASK_RCNN,
    "faster_rcnn": FAST_RCNN,
    "yolo_v4s": YOLO_V4,
    "yolo_v3": YOLO_V3,
}
