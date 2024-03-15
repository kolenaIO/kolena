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
DATASET = "kitti"
TASK = "object-detection-3d"
DEFAULT_DATASET_NAME = f"KITTI [{TASK}]"
ID_FIELDS = ["image_id"]
MODELS = [
    "parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-3class",
    "pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class",
]
