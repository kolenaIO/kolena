# Copyright 2021-2023 Kolena Inc.
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
from typing import List
from typing import Tuple

import numpy as np
from recommender_system.workflow import GroundTruth
from recommender_system.workflow import Inference


def compute_errors(ground_truth: GroundTruth, inference: Inference) -> Tuple[float, float]:
    movie_score_map = {movie.id: movie.score for movie in inference.recommendations}
    rmse = np.sqrt(
        np.mean(
            [
                np.square(movie.score - movie_score_map[movie.id])
                for movie in ground_truth.rated_movies
                if movie.id in movie_score_map.keys()
            ],
        ),
    )
    mae = np.mean(
        [
            np.abs(movie.score - movie_score_map[movie.id])
            for movie in ground_truth.rated_movies
            if movie.id in movie_score_map.keys()
        ],
    )

    return rmse, mae


def precision_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
    relevant_items = len(set(predicted).intersection(actual))
    return relevant_items / k


def recall_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
    relevant_items = len(set(predicted).intersection(actual))
    return relevant_items / len(actual)


def mrr_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
    score = 0.0
    for item in actual:
        rank_q = predicted.index(item) if item in predicted else 0.0
        score += (1.0 / rank_q) if rank_q != 0.0 else 0.0

    return score / len(actual)


def avg_precision_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
    # https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """Order matters"""
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mean_avg_precision_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
    # https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py

    # return np.mean([avg_precision_at_k(actual, predicted, k) for k in range(k)])
    return 0
