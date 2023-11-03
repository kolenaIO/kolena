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
from typing import List, Tuple
import math

import numpy as np
from face_recognition_11.workflow import GroundTruth
from face_recognition_11.workflow import Inference

from kolena.workflow import Histogram


def compute_distance(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    return math.sqrt(math.pow(point_a[0] - point_b[0], 2) + math.pow(point_a[1] - point_b[1], 2))


def compute_distances(
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
) -> Tuple[float, float]:
    distance = compute_distance(point_a, point_b)
    return distance


def calculate_mse_nmse(distances: np.ndarray) -> Tuple[float, float]:
    mse = np.mean(distances**2)
    return mse


def create_similiarity_histogram(
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
) -> Histogram:
    genuine_values = [
        inf.similarity for gt, inf in zip(ground_truths, inferences) if gt.is_same and inf.similarity is not None
    ]

    imposter_values = [
        inf.similarity for gt, inf in zip(ground_truths, inferences) if not gt.is_same and inf.similarity is not None
    ]

    min_data = min(min(genuine_values), min(imposter_values))
    max_data = max(max(genuine_values), max(imposter_values))

    number_of_bins = 50
    bin_size = (max_data - min_data) / number_of_bins
    bin_edges = [min_data + i * bin_size for i in range(number_of_bins + 1)]

    genuine_hist_adjusted = list(np.histogram(genuine_values, bins=bin_edges, density=True)[0])
    imposter_hist_adjusted = list(np.histogram(imposter_values, bins=bin_edges, density=True)[0])

    # histogram of the relative distribution of genuine and imposter pairs, bucketed by similarity score.
    similarity_dist = Histogram(
        title="Similarity Distribution",
        x_label="Similarity Score",
        y_label="Frequency (%)",
        buckets=list(bin_edges),
        frequency=list([genuine_hist_adjusted, imposter_hist_adjusted]),
        labels=["Genuine Pairs", "Imposter Pairs"],
    )

    return similarity_dist
