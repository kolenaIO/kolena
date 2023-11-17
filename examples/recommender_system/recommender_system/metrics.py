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


class ClassificationMetrics:
    def __init__(self, precision_at_k, recall_at_k, tp, fp, fn, tn):
        self.pk = precision_at_k
        self.rk = recall_at_k
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn


def compute_errors(ground_truth: GroundTruth, inference: Inference, k: int) -> Tuple[float, float]:
    predictions = inference.recommendations
    if len(predictions) > k:
        predictions = predictions[:k]

    movie_score_map = {movie.id: movie.score for movie in predictions}
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


def compute_classification_metrics(ground_truth: GroundTruth, inference: Inference, threshold: int, k: int) -> float:
    rated_movies = ground_truth.rated_movies
    recommendations = inference.recommendations

    if len(recommendations) > k:
        recommendations = recommendations[:k]

    ratings = {movie.id for movie in rated_movies}
    predictions = {movie.id for movie in recommendations}

    relevant_movies = len(predictions.intersection(ratings))
    precision_at_k = relevant_movies / k
    recall_at_k = relevant_movies / len(ratings)

    if len(rated_movies) > k:
        rated_movies = rated_movies[:k]

    liked = {movie.id for movie in rated_movies if movie.score >= threshold}
    not_liked = {movie.id for movie in rated_movies if movie.score < threshold}

    tp = len(predictions.intersection(liked))  # recommended & liked
    fp = len(predictions.intersection(not_liked))  # recommended & not liked
    fn = len(liked.difference(predictions))  # not recommended & liked
    tn = k - (tp + fp + fn)  # not recommended & not liked

    return ClassificationMetrics(precision_at_k, recall_at_k, tp, fp, fn, tn)


def mrr(ratings: List[int], predictions: List[int], k: int) -> float:
    if len(predictions) > k:
        predictions = predictions[:k]

    score = 0.0
    for item in ratings:
        rank_q = predictions.index(item) if item in predictions else 0.0
        score += (1.0 / rank_q) if rank_q != 0.0 else 0.0

    return score / len(ratings)


def avg_precision_at_k(ratings: List[int], predictions: List[int], k: int) -> float:
    # https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    if len(predictions) > k:
        predictions = predictions[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predictions):
        if p in ratings and p not in predictions[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not ratings:
        return 0.0

    return score / min(len(ratings), k)


def mean_avg_precision_at_k(ratings: List[int], predictions: List[int], k: int) -> float:
    # https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    return np.mean([avg_precision_at_k(ratings, predictions, k) for k in range(1, k)])
