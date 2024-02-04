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
import math
from typing import List
from typing import Set
from typing import Tuple

import numpy as np
from face_recognition_11.workflow import GroundTruth
from face_recognition_11.workflow import Inference
from face_recognition_11.workflow import TestSample
from face_recognition_11.workflow import TestSampleMetrics

from kolena.workflow import Histogram


class PairMetrics:
    def __init__(
        self,
        n_genuine_pairs: int,
        n_imposter_pairs: int,
        n_fm: int,
        n_fnm: int,
        n_pair_failures: int,
        n_fte: int,
    ) -> None:
        self.genuine_pairs = n_genuine_pairs
        self.imposter_pairs = n_imposter_pairs
        self.fm = n_fm
        self.fmr = n_fm / n_imposter_pairs
        self.fnm = n_fnm
        self.fnmr = n_fnm / n_genuine_pairs
        self.pair_failures = n_pair_failures
        self.fte = n_fte


def compute_threshold(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    fmr: float,
    eps: float = 1e-9,
) -> float:
    # address duplicates
    scores = [
        (match, similarity)
        for gt, inf in zip(ground_truths, inferences)
        for match, similarity in zip(gt.matches, inf.similarities)
    ]
    imposter_scores = sorted(
        [similarity if similarity is not None else 0.0 for match, similarity in scores if not match],
        reverse=True,
    )
    threshold_idx = int(round(fmr * len(imposter_scores) / 2) - 1)
    threshold = imposter_scores[threshold_idx * 2] - eps
    return threshold


def compute_baseline_thresholds(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    lower_range: int,
    upper_range: int,
    num_thresholds: int,
) -> List[Tuple[float, float]]:
    baseline_fmr_x = list(np.logspace(lower_range, upper_range, num_thresholds))
    baseline_thresholds = [compute_threshold(test_samples, ground_truths, inferences, fmr) for fmr in baseline_fmr_x]
    return list(zip(baseline_fmr_x, baseline_thresholds))


def get_unique_pairs(test_samples: List[TestSample]) -> Set[Tuple[str, str]]:
    pairs = {(ts.locator, pair.locator) for ts in test_samples for pair in ts.pairs}
    unique_pairs = {(a, b) if a <= b else (b, a) for a, b in pairs}
    return unique_pairs


def compute_pair_metrics(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    metrics: List[TestSampleMetrics],
) -> PairMetrics:
    unique_pairs = get_unique_pairs(test_samples)
    genuine_pairs, imposter_pairs, fm, fnm, pair_failures, fte = {}, {}, {}, {}, {}, {}

    for ts, gt in zip(test_samples, ground_truths):
        for pair, match in zip(ts.pairs, gt.matches):
            ab = (ts.locator, pair.locator)
            genuine_pairs[ab] = genuine_pairs[ab[::-1]] = match
            imposter_pairs[ab] = imposter_pairs[ab[::-1]] = not match

    for ts, tsm in zip(test_samples, metrics):
        for pair, tsm_pair in zip(ts.pairs, tsm.pair_samples):
            ab = (ts.locator, pair.locator)
            fm[ab] = fm[ab[::-1]] = tsm_pair.is_false_match
            fnm[ab] = fnm[ab[::-1]] = tsm_pair.is_false_non_match
            pair_failures[ab] = pair_failures[ab[::-1]] = tsm_pair.failure_to_enroll
            fte[ab] = fte[ab[::-1]] = (
                tsm.bbox_failure_to_enroll or tsm.keypoint_failure_to_align or tsm_pair.failure_to_enroll
            )

    n_genuine_pairs = np.sum([genuine_pairs[pair] for pair in unique_pairs])
    n_imposter_pairs = np.sum([imposter_pairs[pair] for pair in unique_pairs])
    n_fm = np.sum([fm[pair] for pair in unique_pairs])
    n_fnm = np.sum([fnm[pair] for pair in unique_pairs])
    n_pair_failures = np.sum([pair_failures[pair] for pair in unique_pairs])
    n_fte = np.sum([fte[pair] for pair in unique_pairs])

    return PairMetrics(n_genuine_pairs, n_imposter_pairs, n_fm, n_fnm, n_pair_failures, n_fte)


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

    freq, bin_edges = np.histogram(ious, bins=bin_edges, density=True)  # type: ignore

    return Histogram(
        title="Bounding Box Detection: IoU Distribution",
        x_label="Intersection over Union (IoU)",
        y_label="Frequency (%)",
        buckets=list(bin_edges),
        frequency=list(freq),
    )


def create_similarity_histogram(
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
    genuine_values = np.unique(genuine_values)  # type: ignore
    imposter_values = np.unique(imposter_values)  # type: ignore

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
