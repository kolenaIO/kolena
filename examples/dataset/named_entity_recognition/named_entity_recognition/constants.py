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
from typing import Any
from typing import Dict

DATASET = "n2c2-2014"
MODELS = ["roberta", "bert"]
MODEL_INFO: Dict[str, Dict[str, Any]] = {
    "roberta": {"name": "obi/deid_roberta_i2b2"},
    "bert": {"name": "obi/deid_bert_i2b2", "max_length": 512},
}
