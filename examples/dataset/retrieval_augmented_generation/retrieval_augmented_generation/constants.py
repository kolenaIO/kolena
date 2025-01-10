# Copyright 2021-2025 Kolena Inc.
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
S3_BUCKET = "s3://kolena-public-examples"
DATASET = "financebench"
TASK = "retrieval-augmented_generation"
ID_FIELDS = ["financebench_id"]
MODEL_NAME = {
    "baseline": "gpt-4o-baseline",
    "qme": "gpt-4o-qme",
    "query_decomp": "gpt-4o-qme-query-decomp",
}
