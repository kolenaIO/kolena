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
import boto3
import botocore
from botocore.client import Config


BUCKET = "kolena-public-examples"
DATASET = "JAAD"
ID_FIELDS = ["locator"]
MODELS = [
    "c3d_sort",
    "c3d_deepsort",
    "static_sort",
    "static_deepsort",
]
DEFAULT_DATASET_NAME = "JAAD [crossing-pedestrian-detection]"
VIDEO_FRAME_RATE = 29.97
TRANSPORT_PARAMS = {"client": boto3.client("s3", config=Config(signature_version=botocore.UNSIGNED))}
