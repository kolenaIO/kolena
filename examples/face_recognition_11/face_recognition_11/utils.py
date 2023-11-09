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
from typing import List, Tuple, Callable
import math

import numpy as np
from face_recognition_11.workflow import GroundTruth
from face_recognition_11.workflow import Inference
from face_recognition_11.workflow import TestSampleMetrics, TestSample

from kolena.workflow import Histogram


def compute_threshold(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    fmr: float,
    eps: float = 1e-9,
) -> float:
    func = lambda is_same, similarity, pair_sample: (is_same, similarity)
    scores = filter_duplicate(func, test_samples, ground_truths, inferences)
    imposter_scores = sorted(
        [similarity if similarity is not None else 0.0 for match, similarity in scores if not match],
        reverse=True,
    )
    threshold_idx = int(round(fmr * len(imposter_scores)) - 1)
    threshold = imposter_scores[threshold_idx] - eps

    # print(f"imposter_scores length: {len(imposter_scores)}")
    # print(f"threshold: {threshold}")
    # print(f"threshold_idx: {threshold_idx}")
    return threshold


def filter_duplicate(
    func: Callable,
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    metrics: List[TestSampleMetrics] = None,
) -> List:
    # address duplicates because of how we model
    values = []
    seen = []
    for i, (gt, inf) in enumerate(zip(ground_truths, inferences)):
        for j, (is_same, similarity) in enumerate(zip(gt.matches, inf.similarities)):
            pair = (test_samples[i].locator, test_samples[i].pairs[j].locator)
            if pair not in seen:
                values.append(func(is_same, similarity, metrics[i].pair_samples[j] if metrics is not None else None))
                seen.append(pair)
                seen.append(pair[::-1])

    return values


def compute_distance(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    return math.sqrt(math.pow(point_a[0] - point_b[0], 2) + math.pow(point_a[1] - point_b[1], 2))


def compute_distances(
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
    normalization_factor: float,
) -> Tuple[float, float]:
    distance = compute_distance(point_a, point_b)
    return distance, distance / normalization_factor


def calculate_mse_nmse(distances: np.ndarray, normalization_factor: float) -> Tuple[float, float]:
    mse = np.mean(distances**2)
    nmse = math.sqrt(np.mean((distances / normalization_factor) ** 2))
    return mse, nmse


def create_iou_histogram(
    metrics: List[TestSampleMetrics],
) -> Histogram:
    ious = [tsm.bbox_IoU for tsm in metrics]
    min_data, max_data = 0.0, 1.0

    number_of_bins = 50
    bin_size = (max_data - min_data) / number_of_bins
    bin_edges = [min_data + i * bin_size for i in range(number_of_bins + 1)]

    freq, bin_edges = np.histogram(ious, bins=bin_edges, density=True)

    return Histogram(
        title="Bounding Box Detection: IoU Distribution",
        x_label="Intersection over Union (IoU)",
        y_label="Frequency (%)",
        buckets=list(bin_edges),
        frequency=list(freq),
    )


def create_similarity_histogram(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
) -> Histogram:
    genuine_values = [
        similarity
        for gt, inf in zip(ground_truths, inferences)
        for is_same, similarity in zip(gt.matches, inf.similarities)
        if is_same and similarity is not None
    ]
    imposter_values = [
        similarity
        for gt, inf in zip(ground_truths, inferences)
        for is_same, similarity in zip(gt.matches, inf.similarities)
        if not is_same and similarity is not None
    ]

    # address duplicates
    genuine_values = np.unique(genuine_values)
    imposter_values = np.unique(imposter_values)

    min_data, max_data = 0.0, 1.0

    number_of_bins = 50
    bin_size = (max_data - min_data) / number_of_bins
    bin_edges = [min_data + i * bin_size for i in range(number_of_bins + 1)]

    genuine_hist_adjusted = list(np.histogram(genuine_values, bins=bin_edges, density=True)[0])
    imposter_hist_adjusted = list(np.histogram(imposter_values, bins=bin_edges, density=True)[0])

    # histogram of the relative distribution of genuine and imposter pairs, bucketed by similarity score.
    similarity_dist = Histogram(
        title="Recognition: Similarity Distribution",
        x_label="Similarity Score",
        y_label="Frequency (%)",
        buckets=list(bin_edges),
        frequency=list([genuine_hist_adjusted, imposter_hist_adjusted]),
        labels=["Genuine Pairs", "Imposter Pairs"],
    )

    return similarity_dist
