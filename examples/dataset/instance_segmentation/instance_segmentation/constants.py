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
BUCKET = "kolena-public-examples"
DATASET = "coco-2014-val"
TASK = "transportation"
ID_FIELDS = ["locator"]
MODELS = [
    "yolo_r",
    "yolo_x",
    "yolo_v3",
    "yolo_v4s",
    "faster_rcnn",
    "mask_rcnn",
]
LABEL_TO_COLOR = {
    "bicycle": "#f2f542",
    "bus": "#f542bc",
    "car": "#4260f5",
    "truck": "#8b97a6",
    "train": "#f0f1f2",
    "fire hydrant": "#c542f5",
    "motorcycle": "#66f542",
    "stop sign": "#f54242",
    "traffic light": "#f5aa42",
}
