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
from typing import Any
from typing import Dict
from typing import List

import numpy as np
from keypoint_detection.utils import calculate_mse_nmse
from keypoint_detection.utils import compute_distances

from kolena.workflow.annotation import Keypoints


def compute_metrics(gt: Keypoints, inf: List[Keypoints], norm: float) -> Dict[str, Any]:
    best_face_nmse, best_face_metrics = math.inf, dict(outcome="failure_to_detect")
    for face in inf:
        Δ_nose, norm_Δ_nose = compute_distances(gt.points[0], face.points[0], norm)
        Δ_left_eye, norm_Δ_left_eye = compute_distances(gt.points[1], face.points[1], norm)
        Δ_right_eye, norm_Δ_right_eye = compute_distances(gt.points[2], face.points[2], norm)
        Δ_left_mouth, norm_Δ_left_mouth = compute_distances(gt.points[3], face.points[3], norm)
        Δ_right_mouth, norm_Δ_right_mouth = compute_distances(gt.points[4], face.points[4], norm)
        distances = np.array([Δ_left_eye, Δ_right_eye, Δ_nose, Δ_left_mouth, Δ_right_mouth])
        mse, nmse = calculate_mse_nmse(distances, norm)
        if nmse < best_face_nmse:
            best_face_nmse = nmse
            best_face_metrics = dict(
                outcome="failure_to_align" if nmse > 0.1 else "success",  # TODO: config
                Δ_nose=Δ_nose,
                Δ_left_eye=Δ_left_eye,
                Δ_right_eye=Δ_right_eye,
                Δ_left_mouth=Δ_left_mouth,
                Δ_right_mouth=Δ_right_mouth,
                normalization_factor=norm,
                norm_Δ_nose=norm_Δ_nose,
                norm_Δ_left_eye=norm_Δ_left_eye,
                norm_Δ_right_eye=norm_Δ_right_eye,
                norm_Δ_left_mouth=norm_Δ_left_mouth,
                norm_Δ_right_mouth=norm_Δ_right_mouth,
                mse=mse,
                nmse=nmse,
                best_face=face,
            )
    return best_face_metrics
