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

import numpy as np


def precision_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
    if len(predicted) > k:
        predicted = predicted[:k]

    relevant_items = len(set(predicted).intersection(actual))
    return relevant_items / k


def recall_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
    if len(predicted) > k:
        predicted = predicted[:k]

    relevant_items = len(set(predicted).intersection(actual))
    return relevant_items / len(actual)


def mrr_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    for item in actual:
        rank_q = predicted.index(item) if item in predicted else 0
        score += 1 / rank_q

    return score / len(actual)


def avg_precision_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
    # https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
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
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([avg_precision_at_k(a, p, k) for a, p in zip(actual, predicted)])
