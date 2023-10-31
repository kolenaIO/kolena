from typing import List
import numpy as np


def pk(actual: List[int], predicted: List[int], k: int = 10):
    if len(predicted) > k:
        predicted = predicted[:k]

    relevant_items = len(set(predicted).intersection(actual))
    return relevant_items / k


def apk(actual: List[int], predicted: List[int], k: int = 10):
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


def mapk(actual: List[int], predicted: List[int], k: int = 10):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
