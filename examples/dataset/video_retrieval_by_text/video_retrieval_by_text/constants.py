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
BUCKET = "s3://kolena-public-examples/VATEX/v1.1"
RESULTS = f"{BUCKET}/results"
DATASET_URI = f"{BUCKET}/vatex_metadata.csv"
S3_STORAGE_OPTIONS = {"anon": True}

DEFAULT_DATASET_NAME = "VATEX"
ID_FIELDS = ["caption_id"]
MODELS = ["CLIP-ViTB32", "CLIP2Video-MSRVTT9k", "CLIP2Video-VATEX"]
