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
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from face_recognition_11.image import FRImageAsset

from kolena.annotation import BoundingBox
from kolena.annotation import Keypoints
from kolena.workflow.metrics import iou


def compute_distance(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    return math.sqrt(math.pow(point_a[0] - point_b[0], 2) + math.pow(point_a[1] - point_b[1], 2))


def compute_detection_metrics(
    gt: BoundingBox,
    inf: Optional[BoundingBox],
    configuration: Dict[str, Any],
) -> Dict[str, Any]:
    if inf is None:
        return dict(
            detection_TP=[],
            detection_FP=[],
            detection_FN=[gt],
            detection_is_TP=False,
            detection_is_FP=False,
            detection_is_FN=True,
            detection_failure=True,
            detection_threshold=configuration["iou_threshold"],
        )

    iou_value = iou(gt, inf)
    detected = iou_value >= configuration["iou_threshold"]

    return dict(
        detection_TP=[inf] if detected else [],
        detection_FP=[inf] if not detected else [],
        detection_FN=[gt] if not detected else [],
        detection_is_TP=detected,
        detection_is_FP=not detected,
        detection_is_FN=not detected,
        detection_failure=not detected,
        detection_iou=iou_value,
        detection_threshold=configuration["iou_threshold"],
    )


def compute_alignment_metrics(
    norm_factor: float,
    gt: Keypoints,
    inf: Keypoints,
    configuration: Dict[str, Any],
) -> Dict[str, Any]:
    if len(inf.points) == 0:
        return dict(
            alignment_failure=True,
            alignment_threshold=configuration["nrmse_threshold"],
        )

    Δ_left_eye = compute_distance(gt.points[0], inf.points[0])
    Δ_right_eye = compute_distance(gt.points[1], inf.points[1])
    distances = np.array([Δ_left_eye, Δ_right_eye])
    mse = np.mean(distances**2)
    nrmse = math.sqrt(np.mean((distances / norm_factor) ** 2))

    return dict(
        alignment_MSE=mse,
        alignment_NRMSE=nrmse,
        alignment_Δ_left_eye=Δ_left_eye,
        alignment_Δ_right_eye=Δ_right_eye,
        alignment_norm_Δ_left_eye=Δ_left_eye / norm_factor,
        alignment_norm_Δ_right_eye=Δ_right_eye / norm_factor,
        alignment_failure=nrmse >= configuration["nrmse_threshold"],
        alignment_threshold=configuration["nrmse_threshold"],
    )


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


def compute_pairwise_recognition_merics(is_match: bool, similarity: float, threshold: float) -> Dict[str, Any]:
    predicted_match = similarity > threshold
    return dict(
        is_failure=False,
        is_TM=is_match and predicted_match,
        is_TNM=not is_match and not predicted_match,
        is_FNM=is_match and not predicted_match,
        is_FM=not is_match and predicted_match,
    )


def compute_recognition_merics(pairs: List[FRImageAsset], threshold: float) -> Dict[str, Any]:
    return dict(
        recognition_count_TM=sum(pair.is_TM for pair in pairs if pair.is_TM is not None),
        recognition_count_TNM=sum(pair.is_TNM for pair in pairs if pair.is_TNM is not None),
        recognition_count_FM=sum(pair.is_FM for pair in pairs if pair.is_FM is not None),
        recognition_count_FNM=sum(pair.is_FNM for pair in pairs if pair.is_FNM is not None),
        recognition_FMR=sum(pair.is_FM for pair in pairs if pair.is_FM is not None),
        recognition_FNMR=sum(pair.is_FNM for pair in pairs if pair.is_FNM is not None),
        recognition_threshold=threshold,
        recognition_genuine_similarity=np.mean([pair.similarity for pair in pairs if pair.is_match]),
        recognition_imposter_similarity=np.mean([pair.similarity for pair in pairs if not pair.is_match]),
    )
