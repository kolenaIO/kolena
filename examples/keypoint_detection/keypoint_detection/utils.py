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
