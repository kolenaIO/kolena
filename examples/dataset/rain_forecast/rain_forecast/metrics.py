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


def compute_metrics(ground_truth: str, inference: float, threshold: float = 0.5) -> Dict[str, Any]:
    metrics: Dict[str, Any] = dict(
        missing_ground_truth=True,
        is_correct=None,
        is_tp=None,
        is_fp=None,
        is_fn=None,
        is_tn=None,
    )

    if ground_truth == "Yes" or ground_truth == "No":
        gt = ground_truth == "Yes"
        inf = inference >= threshold
        metrics = dict(
            missing_ground_truth=False,
            is_correct=gt == inf,
            is_tp=gt == inf and gt,
            is_fp=gt != inf and not gt,
            is_fn=gt != inf and gt,
            is_tn=gt == inf and not gt,
        )
        return metrics

    return metrics
