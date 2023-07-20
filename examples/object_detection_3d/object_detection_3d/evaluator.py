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
from dataclasses import dataclass
from enum import IntEnum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import numpy as np
from object_detection_3d.vendored.kitti_eval import _prepare_data

from .utils import ground_truth_to_kitti_format
from .utils import inference_to_kitti_format
from .vendored.kitti_eval import calculate_iou_partly
from .vendored.kitti_eval import compute_statistics_jit
from .vendored.kitti_eval import kitti_eval
from .workflow import GroundTruth
from .workflow import Inference
from .workflow import TestCase
from .workflow import TestSample
from kolena.workflow import Evaluator
from kolena.workflow import EvaluatorConfiguration
from kolena.workflow import MetricsTestCase as BaseMetricsTestCase
from kolena.workflow import MetricsTestSample as BaseMetricsTestSample
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import LabeledBoundingBox3D
from kolena.workflow.annotation import ScoredClassificationLabel
from kolena.workflow.annotation import ScoredLabeledBoundingBox
from kolena.workflow.annotation import ScoredLabeledBoundingBox3D
from kolena.workflow.plot import Curve
from kolena.workflow.plot import CurvePlot
from kolena.workflow.plot import Plot


VALID_LABELS = ["Car", "Pedestrian", "Cyclist"]


class KITTIDifficulty(IntEnum):
    """
    SOURCE: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d

    For cars we require an 3D bounding box overlap of 70%,
    while for pedestrians and cyclists we require a 3D bounding box overlap of 50%.

    Difficulties are defined as follows:

    Easy: Min. bounding box height: 40 Px, Max. occlusion level: Fully visible, Max. truncation: 15 %
    Moderate: Min. bounding box height: 25 Px, Max. occlusion level: Partly occluded, Max. truncation: 30 %
    Hard: Min. bounding box height: 25 Px, Max. occlusion level: Difficult to see, Max. truncation: 50 %
    """

    EASY = 0
    MODERATE = 1
    HARD = 2


@dataclass(frozen=True)
class KITTI3DConfig(EvaluatorConfiguration):
    difficulty: KITTIDifficulty

    def display_name(self):
        return f"KITTI Difficulty ({self.name()})"

    def name(self):
        if self.difficulty == KITTIDifficulty.EASY:
            name = "easy"
        elif self.difficulty == KITTIDifficulty.MODERATE:
            name = "moderate"
        else:
            name = "hard"
        return name


@dataclass(frozen=True)
class UnmatchedLabeledBoundingBox3D(LabeledBoundingBox3D):
    max_overlap: float

    @staticmethod
    def from_bbox(bbox: LabeledBoundingBox3D, overlap: float) -> "UnmatchedLabeledBoundingBox3D":
        return UnmatchedLabeledBoundingBox3D(
            label=bbox.label,
            center=bbox.center,
            dimensions=bbox.dimensions,
            rotations=bbox.rotations,
            max_overlap=overlap,
        )


@dataclass(frozen=True)
class UnmatchedScoredBoundingBox3D(ScoredLabeledBoundingBox3D):
    max_overlap: float

    @staticmethod
    def from_bbox(bbox: ScoredLabeledBoundingBox3D, overlap: float) -> "UnmatchedScoredBoundingBox3D":
        return UnmatchedScoredBoundingBox3D(
            label=bbox.label,
            score=bbox.score,
            center=bbox.center,
            dimensions=bbox.dimensions,
            rotations=bbox.rotations,
            max_overlap=overlap,
        )


@dataclass(frozen=True)
class MetricsTestSample(BaseMetricsTestSample):
    nInferences: int
    nValidObjects: int
    thresholds: List[ScoredClassificationLabel]
    nMatchedInferences: int
    nMissedObjects: int
    nMismatchedInferences: int
    FP_2D: List[ScoredLabeledBoundingBox]
    FP_3D: List[UnmatchedScoredBoundingBox3D]
    FN_2D: List[LabeledBoundingBox]
    FN_3D: List[UnmatchedLabeledBoundingBox3D]
    TP_2D: List[ScoredLabeledBoundingBox]
    TP_3D: List[ScoredLabeledBoundingBox3D]


@dataclass(frozen=True)
class MetricsTestCaseLabel(BaseMetricsTestCase):
    label: str
    nObjects: int
    nInferences: int
    mAP_2D: float
    mAP_3D: float
    mAP_BEV: float


@dataclass(frozen=True)
class MetricsTestCase(BaseMetricsTestCase):
    per_label: List[MetricsTestCaseLabel]
    nObjects: int
    nInferences: int
    mAP_2D_macro: float
    mAP_3D_macro: float
    mAP_BEV_macro: float


class KITTI3DEvaluator(Evaluator):
    metrics_by_test_case: Dict[str, Dict[str, float]] = {}

    def get_test_case_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
    ) -> Dict[str, Union[float, np.ndarray]]:
        if test_case.name not in self.metrics_by_test_case.keys():
            self.metrics_by_test_case[test_case.name] = self.evaluate(inferences)

        return self.metrics_by_test_case[test_case.name]

    def compute_test_sample_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        configuration: Optional[KITTI3DConfig] = None,
    ) -> List[Tuple[TestSample, MetricsTestSample]]:
        sample_metrics: List[Tuple[TestSample, MetricsTestSample]] = []
        if configuration is None:
            raise ValueError(f"{type(self).__name__} must have configuration")

        inferences = sorted(inferences, key=lambda x: x[0].locator)
        test_case_metrics = self.get_test_case_metrics(test_case, inferences)
        f1_optimal_thresholds = {}
        min_overlaps = [0.7, 0.5, 0.5]
        gt_annos = [ground_truth_to_kitti_format(sample, gt) for sample, gt, _ in inferences]
        dt_annos = [inference_to_kitti_format(sample, inf) for sample, _, inf in inferences]
        difficulty_name = ["easy", "moderate", "hard"]
        class_name_value = {"Car": 0, "Cyclist": 2, "Pedestrian": 1}
        difficulty = configuration.difficulty.value
        results = [{} for _ in range(len(inferences))]
        overlaps, parted_overlaps, total_dt_num, total_gt_num = calculate_iou_partly(dt_annos, gt_annos, 2, 200)
        ignored_gts_combined = [[True] * len(gt.bboxes_2d) for _, gt, inf in inferences]

        for current_class in VALID_LABELS:
            prefix = f"bbox_{current_class}_{difficulty_name[difficulty]}"
            precisions = test_case_metrics[f"{prefix}_precisions"]
            recalls = test_case_metrics[f"{prefix}_recalls"]
            f1 = [2 * precision * recall / (precision + recall) for precision, recall in zip(precisions, recalls)]
            threshold = np.max(f1)
            f1_optimal_thresholds[current_class] = threshold
            class_value = class_name_value[current_class]

            rets = _prepare_data(gt_annos, dt_annos, class_value, difficulty)
            (
                gt_datas_list,
                dt_datas_list,
                ignored_gts,
                ignored_dets,
                dontcares,
                total_dc_num,
                total_num_valid_gt,
            ) = rets
            for i, ignored_gt in enumerate(ignored_gts):
                for j, ignore in enumerate(ignored_gt):
                    if ignore == 0:
                        ignored_gts_combined[i][j] = False
            for i in range(len(gt_annos)):
                tp, fp, fn, similarity, thresholds, tps, fps, fns = compute_statistics_jit(
                    overlaps[i],
                    gt_datas_list[i],
                    dt_datas_list[i],
                    ignored_gts[i],
                    ignored_dets[i],
                    dontcares[i],
                    "3d",
                    min_overlap=min_overlaps[class_value],
                    thresh=threshold,
                    compute_fp=True,
                    compute_aos=False,
                )
                results[i][current_class] = dict(tp=tps, fp=fps, fn=fns)

        for i, (sample, gt, inf) in enumerate(inferences):
            result = results[i]
            TP = [sum(tp) for tp in zip(result["Car"]["tp"], result["Cyclist"]["tp"], result["Pedestrian"]["tp"])]
            FP = [sum(fp) for fp in zip(result["Car"]["fp"], result["Cyclist"]["fp"], result["Pedestrian"]["fp"])]
            FN = [sum(fn) for fn in zip(result["Car"]["fn"], result["Cyclist"]["fn"], result["Pedestrian"]["fn"])]
            FP_2D = [inf.bboxes_2d[j] for j, fp in enumerate(FP) if fp]
            FP_3D = [
                UnmatchedScoredBoundingBox3D.from_bbox(inf.bboxes_3d[j], np.max(overlaps[i][j]))
                for j, fp in enumerate(FP)
                if fp
            ]
            TP_2D = [inf.bboxes_2d[j] for j, tp in enumerate(TP) if tp]
            TP_3D = [inf.bboxes_3d[j] for j, tp in enumerate(TP) if tp]
            FN_2D = [gt.bboxes_2d[j] for j, fn in enumerate(FN) if fn]
            FN_3D = [
                UnmatchedLabeledBoundingBox3D.from_bbox(gt.bboxes_3d[j], np.max(overlaps[i][:, j]))
                for j, fn in enumerate(FN)
                if fn
            ]
            sample_metrics.append(
                (
                    sample,
                    MetricsTestSample(
                        nInferences=len(inf.bboxes_3d),
                        nValidObjects=sum(1 for ignore in ignored_gts_combined[i] if not ignore),
                        thresholds=[
                            ScoredClassificationLabel(score=score, label=label)
                            for label, score in sorted(f1_optimal_thresholds.items())
                        ],
                        nMatchedInferences=len(TP_2D),
                        nMissedObjects=len(FN_2D),
                        nMismatchedInferences=len(FP_2D),
                        FP_2D=FP_2D,
                        FP_3D=FP_3D,
                        TP_2D=TP_2D,
                        TP_3D=TP_3D,
                        FN_2D=FN_2D,
                        FN_3D=FN_3D,
                    ),
                ),
            )
        return sample_metrics

    def evaluate(
        self,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
    ) -> Dict[str, float]:
        # convert GT & INF to KITTI evaluation suite's compatible type
        kitti_ground_truths: List[Dict[str, Any]] = []
        kitti_inferences: List[Dict[str, Any]] = []
        labels: Set[str] = set()
        target_labels = {"Car", "Pedestrian", "Cyclist"}

        for sample, gt, inf in inferences:
            kitti_ground_truths.append(ground_truth_to_kitti_format(sample, gt))
            kitti_inferences.append(inference_to_kitti_format(sample, inf))
            # get the set of labels used in this test case
            labels.update({bbox.label for bbox in gt.bboxes_2d if bbox.label in target_labels})

        _, metrics = kitti_eval(kitti_ground_truths, kitti_inferences, list(labels), eval_types=["bbox", "bev", "3d"])
        for label in target_labels:
            metrics[f"{label}_num_ground_truths"] = np.sum(
                [sum([1 for bbox in gt.bboxes_2d if bbox.label == label]) for _, gt, _ in inferences],
            )
            metrics[f"{label}_num_inferences"] = np.sum(
                [1 if box.label == label else 0 for _, _, inf in inferences for box in inf.bboxes_2d],
            )
        metrics["num_ground_truths"] = np.sum(
            [sum([1 for bbox in gt.bboxes_2d if bbox.label != "DontCare"]) for _, gt, _ in inferences],
        )
        metrics["num_inferences"] = np.sum(
            [1 if box.label in labels else 0 for _, _, inf in inferences for box in inf.bboxes_2d],
        )
        return metrics

    def compute_test_case_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        metrics: List[MetricsTestSample],
        configuration: Optional[KITTI3DConfig] = None,
    ) -> MetricsTestCase:
        if configuration is None:
            raise ValueError(f"{type(self).__name__} must have configuration")

        test_case_metrics = self.get_test_case_metrics(test_case, inferences)
        return MetricsTestCase(
            per_label=[
                MetricsTestCaseLabel(
                    label=label,
                    nObjects=int(test_case_metrics[f"{label}_num_ground_truths"]),
                    nInferences=int(test_case_metrics[f"{label}_num_inferences"]),
                    mAP_2D=test_case_metrics[f"KITTI/{label}_2D_AP40_{configuration.name()}_strict"],
                    mAP_3D=test_case_metrics[f"KITTI/{label}_3D_AP40_{configuration.name()}_strict"],
                    mAP_BEV=test_case_metrics[f"KITTI/{label}_BEV_AP40_{configuration.name()}_strict"],
                )
                for label in test_case_metrics["classes"]
            ],
            nObjects=int(test_case_metrics["num_ground_truths"]),
            nInferences=int(test_case_metrics["num_inferences"]),
            mAP_2D_macro=test_case_metrics[f"KITTI/Overall_2D_AP40_{configuration.name()}"],
            mAP_3D_macro=test_case_metrics[f"KITTI/Overall_3D_AP40_{configuration.name()}"],
            mAP_BEV_macro=test_case_metrics[f"KITTI/Overall_BEV_AP40_{configuration.name()}"],
        )

    def compute_test_case_plots(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        metrics: List[MetricsTestSample],
        configuration: Optional[KITTI3DConfig] = None,
    ) -> Optional[List[Plot]]:
        if configuration is None:
            raise ValueError(f"{type(self).__name__} must have configuration")

        test_case_metrics = self.get_test_case_metrics(test_case, inferences)
        plots = [
            CurvePlot(
                title="Precision vs. Recall [2D BoundingBox Evaluation]",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(
                        x=test_case_metrics[f"bbox_{class_name}_{configuration.name()}_recalls"].tolist(),
                        y=test_case_metrics[f"bbox_{class_name}_{configuration.name()}_precisions"].tolist(),
                        label=f"{class_name}",
                    )
                    for class_name in test_case_metrics["classes"]
                ],
            ),
            CurvePlot(
                title="Precision vs. Recall [3D BoundingBox Evaluation]",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(
                        x=test_case_metrics[f"3d_{class_name}_{configuration.name()}_recalls"].tolist(),
                        y=test_case_metrics[f"3d_{class_name}_{configuration.name()}_precisions"].tolist(),
                        label=f"{class_name}",
                    )
                    for class_name in test_case_metrics["classes"]
                ],
            ),
            CurvePlot(
                title="Precision vs. Recall [BEV Evaluation]",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(
                        x=test_case_metrics[f"bev_{class_name}_{configuration.name()}_recalls"].tolist(),
                        y=test_case_metrics[f"bev_{class_name}_{configuration.name()}_precisions"].tolist(),
                        label=f"{class_name}",
                    )
                    for class_name in test_case_metrics["classes"]
                ],
            ),
        ]
        return plots
