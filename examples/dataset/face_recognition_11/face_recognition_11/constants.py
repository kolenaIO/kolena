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
DATASET = "labeled-faces-in-the-wild"

DATASET_METADATA = f"s3://{BUCKET}/{DATASET}/raw/metadata.csv"
DATASET_PAIRS = f"s3://{BUCKET}/{DATASET}/raw/pairs.csv"
DATASET_DETECTION = f"s3://{BUCKET}/{DATASET}/raw/detection.csv"

EVAL_CONFIG = dict(
    false_match_rate=1e-1,
    iou_threshold=0.5,
    nrmse_threshold=0.08,
)
