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

import pandas as pd


def compute_recognition_threshold(df_pair_results: pd.DataFrame, fmr: float, eps: float = 1e-9) -> float:
    imposter_scores = sorted(
        [
            pair.similarity if pair.similarity is not None else 0.0
            for pair in df_pair_results.itertuples()
            if not pair.is_match
        ],
        reverse=True,
    )
    threshold_idx = int(round(fmr * len(imposter_scores) / 2) - 1)
    threshold = imposter_scores[threshold_idx * 2] - eps
    return threshold


def compute_pairwise_recognition_metrics(is_match: bool, similarity: float, threshold: float) -> Dict[str, Any]:
    predicted_match = similarity > threshold
    return dict(
        threshold=threshold,
        is_predicted_match=predicted_match,
        is_TM=is_match and predicted_match,
        is_TNM=not is_match and not predicted_match,
        is_FM=not is_match and predicted_match,
        is_FNM=is_match and not predicted_match,
        FMR=None if is_match else 1 if predicted_match else 0,
        FNMR=None if not is_match else 0 if predicted_match else 1,
    )
