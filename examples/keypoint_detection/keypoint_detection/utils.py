import math
import re
from typing import Tuple

import numpy as np


def get_readable(text: str) -> str:
    # no spaces before periods, only after
    return re.sub(r"\s+(\.)", r"\1", text)


def compute_distance(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    return math.sqrt(math.pow(point_a[0] - point_b[0], 2) + math.pow(point_a[1] - point_b[1], 2))


def compute_distances(
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
    normalization_factor: float,
) -> Tuple[float, float]:
    distance = compute_distance(point_a, point_b)
    return distance, distance / normalization_factor


def calculate_mse_nmse(distances: np.ndarray) -> Tuple[float, float]:
    mse = np.mean(distances**2)
    squared_diff_sum = np.sum((distances - np.mean(distances)) ** 2)
    true_squared_sum = np.sum(distances**2)
    nmse = squared_diff_sum / (mse * true_squared_sum)
    return mse, nmse
