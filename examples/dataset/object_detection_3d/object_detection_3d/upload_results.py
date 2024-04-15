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
import dataclasses
from argparse import ArgumentParser
from argparse import Namespace
from typing import Any
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple

import numpy as np
import pandas as pd
from object_detection_3d.constants import BUCKET
from object_detection_3d.constants import DATASET
from object_detection_3d.constants import DEFAULT_DATASET_NAME
from object_detection_3d.constants import ID_FIELDS
from object_detection_3d.constants import MODELS
from object_detection_3d.constants import TASK
from object_detection_3d.utils import alpha_from_bbox3D
from object_detection_3d.utils import bbox_to_kitti_format
from object_detection_3d.utils import center_to_kitti_format
from object_detection_3d.utils import transform_to_camera_frame
from object_detection_3d.vendored.kitti_eval import _prepare_data  # type: ignore[attr-defined]
from object_detection_3d.vendored.kitti_eval import calculate_iou_partly  # type: ignore[attr-defined]
from object_detection_3d.vendored.kitti_eval import compute_statistics_jit  # type: ignore[attr-defined]
from object_detection_3d.vendored.kitti_eval import kitti_eval  # type: ignore[attr-defined]

from kolena import dataset
from kolena.annotation import LabeledBoundingBox3D
from kolena.annotation import ScoredLabeledBoundingBox
from kolena.annotation import ScoredLabeledBoundingBox3D
from kolena.dataset import upload_results

CLASS_NAME_VALUE = {"Car": 0, "Cyclist": 2, "Pedestrian": 1}
VALID_LABELS = ["Car", "Pedestrian", "Cyclist"]
KITTY_DIFFICULTY = {"easy": 0, "moderate": 1, "hard": 2}
KITTY_DIFFICULTY_INV = ["easy", "moderate", "hard"]


def to_kitti_format(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Set[str]]:
    gt_annos = []
    dt_annos = []
    labels = set()
    for record in df.itertuples():
        gt_bboxes_2d = record.image_bboxes
        gt_bboxes_3d = record.velodyne_bboxes
        inf_bboxes_2d = record.raw_inferences_2d
        inf_bboxes_3d = record.raw_inferences_3d
        velo_to_camera = record.velodyne_to_camera_transformation
        camera_rect = record.camera_rectification
        if len(gt_bboxes_3d) == 0:
            gt = dict(
                name=np.array([]),
                truncated=np.array([]),
                occluded=np.array([]),
                alpha=np.array([]),
                bbox=np.zeros([0, 4]),
                dimensions=np.zeros([0, 3]),
                location=np.zeros([0, 3]),
                rotation_y=np.array([]),
            )

        else:
            camera_bboxes = transform_to_camera_frame(
                gt_bboxes_3d,  # type: ignore
                velo_to_camera,
                camera_rect,
            )
            gt_labels = [bbox.label for bbox in gt_bboxes_3d]
            labels.update(gt_labels)
            gt = dict(
                name=np.array(gt_labels),
                truncated=np.array([bbox.truncated for bbox in gt_bboxes_3d]),
                occluded=np.array([bbox.occluded for bbox in gt_bboxes_3d]),
                alpha=np.array(
                    [alpha_from_bbox3D(bbox, bbox_lidar) for bbox, bbox_lidar in zip(camera_bboxes, gt_bboxes_3d)],
                ),
                bbox=np.array([bbox_to_kitti_format(bbox) for bbox in gt_bboxes_2d]),
                dimensions=np.array([bbox.dimensions for bbox in camera_bboxes]),
                location=np.array([center_to_kitti_format(bbox) for bbox in camera_bboxes]),
                rotation_y=np.array([bbox.rotations[1] for bbox in camera_bboxes]),
            )

        if len(inf_bboxes_3d) == 0:
            inf = dict(
                name=np.array([]),
                truncated=np.array([]),
                occluded=np.array([]),
                alpha=np.array([]),
                bbox=np.zeros([0, 4]),
                dimensions=np.zeros([0, 3]),
                location=np.zeros([0, 3]),
                rotation_y=np.array([]),
                score=np.array([]),
            )
        else:
            pred_camera_bboxes = transform_to_camera_frame(
                inf_bboxes_3d,  # type: ignore
                velo_to_camera,
                camera_rect,
            )
            inf_labels = [bbox.label for bbox in inf_bboxes_3d]
            labels.update(inf_labels)
            inf = dict(
                name=np.array(inf_labels),
                truncated=np.array([0.0] * len(inf_bboxes_3d)),
                occluded=np.array([0] * len(inf_bboxes_3d)),
                alpha=np.array(
                    [
                        alpha_from_bbox3D(bbox, bbox_lidar)
                        for bbox, bbox_lidar in zip(pred_camera_bboxes, inf_bboxes_3d)
                    ],
                ),
                bbox=np.array([bbox_to_kitti_format(bbox) for bbox in inf_bboxes_2d]),
                dimensions=np.array([bbox.dimensions for bbox in pred_camera_bboxes]),
                location=np.array([center_to_kitti_format(bbox) for bbox in pred_camera_bboxes]),
                rotation_y=np.array([bbox.rotations[1] for bbox in pred_camera_bboxes]),
                score=np.array([bbox.score for bbox in inf_bboxes_3d]),
            )

        gt_annos.append(gt)
        dt_annos.append(inf)

    return gt_annos, dt_annos, labels


def compute_f1_optimal_thresholds(
    kitti_ground_truths: List[Dict[str, Any]],
    kitti_inferences: List[Dict[str, Any]],
    labels: Set[str],
) -> Dict[int, Dict[str, float]]:
    labels = {label for label in labels if label in VALID_LABELS}
    f1_optimal_thresholds = {}

    _, metrics = kitti_eval(kitti_ground_truths, kitti_inferences, list(labels), eval_types=["bev", "3d"])
    for name, difficulty in KITTY_DIFFICULTY.items():
        thresholds = {}
        for current_class in VALID_LABELS:
            prefix = f"bbox_{current_class}_{name}"
            precisions = metrics[f"{prefix}_precisions"]
            recalls = metrics[f"{prefix}_recalls"]
            f1 = [
                2 * precision * recall / (precision + recall)
                for precision, recall in zip(precisions, recalls)  # type: ignore
            ]
            thresholds[current_class] = np.max(f1)

        f1_optimal_thresholds[difficulty] = thresholds

    return f1_optimal_thresholds


def compute_metrics_by_difficulty(df: pd.DataFrame) -> List[Tuple[Dict[str, Any], pd.DataFrame]]:
    min_overlaps = [0.7, 0.5, 0.5]
    gt_annos, dt_annos, labels = to_kitti_format(df)
    f1_optimal_thresholds = compute_f1_optimal_thresholds(gt_annos, dt_annos, labels)
    overlaps, _, total_dt_num, total_gt_num = calculate_iou_partly(dt_annos, gt_annos, 2, 200)
    ignored_gts_combined = [[True] * len(gt_bboxes) for gt_bboxes in df["image_bboxes"]]

    def compute_metrics_with_difficulty(difficulty: int) -> pd.DataFrame:
        sample_metrics = []
        results: List[Dict[str, Dict[Any, Any]]] = [{} for _ in range(len(df))]
        raw_results: List[Dict[str, Dict[Any, Any]]] = [{} for _ in range(len(df))]
        current_optimal_thresholds = f1_optimal_thresholds[difficulty]

        for current_class in VALID_LABELS:
            threshold = current_optimal_thresholds[current_class]
            class_value = CLASS_NAME_VALUE[current_class]

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
                all_tp, all_fp, all_fn, all_similarity, _, all_tps, all_fps, all_fns = compute_statistics_jit(
                    overlaps[i],
                    gt_datas_list[i],
                    dt_datas_list[i],
                    ignored_gts[i],
                    ignored_dets[i],
                    dontcares[i],
                    "3d",
                    min_overlap=min_overlaps[class_value],
                    compute_fp=True,
                )
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
                raw_results[i][current_class] = dict(tp=all_tps, fp=all_fps, fn=all_fns)

        for i, record in enumerate(df.itertuples()):
            result = results[i]
            raw_result = raw_results[i]
            TP = [sum(tp) for tp in zip(result["Car"]["tp"], result["Cyclist"]["tp"], result["Pedestrian"]["tp"])]
            FP = [sum(fp) for fp in zip(result["Car"]["fp"], result["Cyclist"]["fp"], result["Pedestrian"]["fp"])]
            FN = [sum(fn) for fn in zip(result["Car"]["fn"], result["Cyclist"]["fn"], result["Pedestrian"]["fn"])]
            # FP_2D = [inferences for inferences, fp in zip(record.raw_inferences_2d, FP) if fp]
            FP_3D = [
                dataclasses.replace(
                    record.raw_inferences_3d[j],
                    max_overlap=np.max(overlaps[i][j]),  # type: ignore[call-arg]
                    match_index=np.argmax(overlaps[i][j]),  # type: ignore[call-arg]
                )
                for j, fp in enumerate(FP)
                if fp
            ]
            # TP_2D = [inferences for inferences, tp in zip(record.raw_inferences_2d, TP) if tp]
            TP_3D = [
                ScoredLabeledBoundingBox3D(
                    **record.raw_inferences_3d[j]._to_dict(),
                    overlap=np.max(overlaps[i][j]),  # type: ignore[call-arg]
                    match_index=np.argmax(overlaps[i][j]),  # type: ignore[call-arg]
                )
                for j, tp in enumerate(TP)
                if tp
            ]
            # FN_2D = [image_bboxes for image_bboxes, fn in zip(record.image_bboxes, FN) if fn]
            FN_3D = [
                LabeledBoundingBox3D(
                    **record.velodyne_bboxes[j]._to_dict(),
                    max_overlap=np.max(overlaps[i][:, j]) if len(overlaps[i][:, j]) else None,  # type: ignore[call-arg]
                    match_index=(
                        np.argmax(overlaps[i][:, j]) if len(overlaps[i][:, j]) else None
                    ),  # type: ignore[call-arg]
                )
                for j, fn in enumerate(FN)
                if fn
            ]
            matched_inference = [
                inferences
                for inferences, tps in zip(
                    record.raw_inferences_3d,
                    zip(raw_result["Car"]["tp"], raw_result["Cyclist"]["tp"], raw_result["Pedestrian"]["tp"]),
                )
                if sum(tps)
            ]
            unmatched_inference = [
                inferences
                for inferences, fps in zip(
                    record.raw_inferences_3d,
                    zip(raw_result["Car"]["fp"], raw_result["Cyclist"]["fp"], raw_result["Pedestrian"]["fp"]),
                )
                if sum(fps)
            ]
            unmatched_ground_truth = [
                velodyne_bboxes
                for velodyne_bboxes, fns in zip(
                    record.velodyne_bboxes,
                    zip(raw_result["Car"]["fn"], raw_result["Cyclist"]["fn"], raw_result["Pedestrian"]["fn"]),
                )
                if sum(fns)
            ]

            sample_metrics.append(
                dict(
                    image_id=record.image_id,
                    matched_inference=matched_inference,
                    unmatched_ground_truth=unmatched_ground_truth,
                    unmatched_inference=unmatched_inference,
                    nInferences=len(record.raw_inferences_3d),
                    nValidObjects=sum(1 for ignore in ignored_gts_combined[i] if not ignore),
                    nMatchedInferences=len(TP_3D),
                    nMissedObjects=len(FN_3D),
                    nMismatchedInferences=len(FP_3D),
                    thresholds=current_optimal_thresholds,
                    # FP_2D=FP_2D,
                    FP_3D=FP_3D,
                    # TP_2D=TP_2D,
                    TP_3D=TP_3D,
                    # FN_2D=FN_2D,
                    FN_3D=FN_3D,
                    images_with_inferences=[
                        dataclasses.replace(
                            img,
                            FP_3D=FP_3D,
                            TP_3D=TP_3D,
                            FN_3D=FN_3D,
                            matched_inference=matched_inference,
                            unmatched_ground_truth=unmatched_ground_truth,
                            unmatched_inference=unmatched_inference,
                        )
                        for img in record.images
                    ],
                ),
            )
        return pd.DataFrame(sample_metrics)

    data = []
    gt_info_columns = ["velodyne_to_camera_transformation", "camera_rectification"]
    for name, difficulty in KITTY_DIFFICULTY.items():
        eval_config = dict(
            difficulty=name,
            threshold_strategy="F1-Optimal",
            thresholds=f1_optimal_thresholds[difficulty],
        )
        metrics_df = compute_metrics_with_difficulty(difficulty)
        results_df = df[["image_id", "raw_inferences_2d", "raw_inferences_3d", *gt_info_columns]].merge(
            metrics_df,
            on="image_id",
        )
        results_df.drop(columns=gt_info_columns, inplace=True)
        data.append((eval_config, results_df))

    return data


def _get_box(bboxes: List) -> pd.Series:
    bboxes_2d = [
        ScoredLabeledBoundingBox(
            top_left=(box["box"][0], box["box"][1]),
            bottom_right=(box["box"][2], box["box"][3]),
            score=box["score"],
            label=box["pred"],
        )
        for box in bboxes
    ]
    bboxes_3d = [
        ScoredLabeledBoundingBox3D(
            dimensions=(box["box3d"][3], box["box3d"][4], box["box3d"][5]),
            center=(box["box3d"][0], box["box3d"][1], box["box3d"][2] + (box["box3d"][5] / 2.0)),
            rotations=(0.0, 0.0, box["box3d"][6]),
            score=box["score"],
            label=box["pred"],
        )
        for box in bboxes
    ]

    return pd.Series(
        [bboxes_2d, bboxes_3d],
        index=["raw_inferences_2d", "raw_inferences_3d"],
    )


def load_results(model: str) -> pd.DataFrame:
    df = pd.read_json(
        f"s3://{BUCKET}/{DATASET}/{TASK}/results/raw/{model}.jsonl",
        lines=True,
        dtype=False,
        storage_options={"anon": True},
    )
    df[["raw_inferences_2d", "raw_inferences_3d"]] = df["bboxes"].apply(_get_box)
    return df


def run(args: Namespace) -> None:
    pred_df = load_results(args.model)
    dataset_df = dataset.download_dataset(args.dataset).sort_values(by="image_id", ignore_index=True)
    results_df = dataset_df.merge(pred_df, on=ID_FIELDS)
    results = compute_metrics_by_difficulty(results_df)
    upload_results(args.dataset, args.model, results)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("model", type=str, choices=MODELS, help="Name of the model to test.")
    ap.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Optionally specify a custom dataset name to test.",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
