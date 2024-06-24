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
import pickle
from argparse import ArgumentParser
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import boto3
import botocore
import numpy as np
import pandas as pd
from botocore.client import Config
from crossing_pedestrian_detection.constants import BUCKET
from crossing_pedestrian_detection.constants import DATASET
from crossing_pedestrian_detection.constants import DEFAULT_DATASET_NAME
from crossing_pedestrian_detection.constants import MODELS
from crossing_pedestrian_detection.utils import FrameMatch
from crossing_pedestrian_detection.utils import PedestrianBoundingBox
from crossing_pedestrian_detection.utils import process_ped_annotations
from crossing_pedestrian_detection.utils import ScoredPedestrianBoundingBox
from pydantic.dataclasses import dataclass
from smart_open import open as smart_open

from kolena.annotation import BoundingBox
from kolena.dataset import download_dataset
from kolena.dataset import upload_results
from kolena.metrics import f1_score
from kolena.metrics import match_inferences
from kolena.metrics import precision
from kolena.metrics import recall

TRANSPORT_PARAMS = {"client": boto3.client("s3", config=Config(signature_version=botocore.UNSIGNED))}

THRESHOLD = 0.5
CONFIDENCE = 0.01


@dataclass(frozen=True)
class FrameMetrics:
    TP: List[ScoredPedestrianBoundingBox]
    FP: List[ScoredPedestrianBoundingBox]
    FN: List[BoundingBox]
    TN: List[BoundingBox]


def postprocess_inferences(inferences: List[ScoredPedestrianBoundingBox]) -> List[ScoredPedestrianBoundingBox]:
    n_observation_frames = 16
    in_observation_state = True
    previous_conf = -1.0
    processed_inferences = []
    for i, inf in enumerate(inferences):
        confidence_score = inf.score
        if in_observation_state and i > n_observation_frames:
            in_observation_state = False

        if inf.failed_to_infer:
            if previous_conf >= 0.0:
                confidence_score = previous_conf
            else:
                confidence_score = 0.0
        else:
            if in_observation_state:
                in_observation_state = False
                if inf.score is not None:
                    previous_conf = inf.score
            else:
                if inf.score is not None:
                    previous_conf = inf.score
        fail_inf = inf.failed_to_infer if not in_observation_state and previous_conf < 0 else False
        processed_inferences.append(
            ScoredPedestrianBoundingBox(
                ped_id=inf.ped_id,
                top_left=inf.top_left,
                bottom_right=inf.bottom_right,
                frame_id=inf.frame_id,
                occlusion=inf.occlusion,
                time_to_event=inf.time_to_event,
                failed_to_infer=fail_inf,
                score=confidence_score,
                observation=in_observation_state,
                label=inf.label,
            ),
        )

    return processed_inferences


def process_inf_data(action_model_name: str, detection_model_name: str) -> Dict[str, List[ScoredPedestrianBoundingBox]]:
    model_pkl_name = f"s3://{BUCKET}/{DATASET}/raw/jaad_{action_model_name}_{detection_model_name}_database.pkl"
    with smart_open(model_pkl_name, "rb", transport_params=TRANSPORT_PARAMS) as inf_file:
        inf_annotations = pickle.load(inf_file)

    results_pkl = f"s3://{BUCKET}/{DATASET}/results/raw/{action_model_name}_{detection_model_name}.pkl"
    with smart_open(results_pkl, "rb", transport_params=TRANSPORT_PARAMS) as results_file:
        results = pickle.load(results_file)
        predictions: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        for pid, tte, y, img in zip(results["pid"], results["tte"], results["y"], results["image"]):
            pid = pid[0][0]
            frame_id = int(Path(img[0]).stem)
            predictions[pid][frame_id]["tte"] = tte[0]
            predictions[pid][frame_id]["confidence"] = y[0]

    scored_bboxes_per_ped: Dict[str, List[ScoredPedestrianBoundingBox]] = {}
    for filename in inf_annotations:
        bboxes_per_file = process_ped_annotations(inf_annotations[filename]["ped_annotations"])
        scored_bboxes_lst = []
        for ped_id, bboxes in bboxes_per_file.items():
            for bbox in bboxes:
                if bbox.ped_id not in predictions or bbox.frame_id not in predictions[ped_id]:
                    failed_to_infer = True
                else:
                    failed_to_infer = False
                    confidence = predictions[bbox.ped_id][bbox.frame_id]["confidence"]
                    time_to_event = predictions[bbox.ped_id][bbox.frame_id]["tte"]

                scored_bboxes_lst.append(
                    ScoredPedestrianBoundingBox(
                        ped_id=bbox.ped_id,
                        top_left=bbox.top_left,
                        bottom_right=bbox.bottom_right,
                        frame_id=bbox.frame_id,
                        occlusion=bbox.occlusion,
                        time_to_event=time_to_event if not failed_to_infer else None,
                        failed_to_infer=failed_to_infer,
                        score=confidence if not failed_to_infer else 0,
                        label=bbox.label,
                    ),
                )
        scored_bboxes_per_ped[filename] = postprocess_inferences(scored_bboxes_lst)

    return scored_bboxes_per_ped


def compute_pedestrian_metrics(
    gt_bboxes: List[PedestrianBoundingBox],
    inference_bboxes: List[ScoredPedestrianBoundingBox],
) -> Dict[int, FrameMatch]:
    gt_bbox_per_frame = {bbox.frame_id: bbox for bbox in gt_bboxes}
    inf_bboxes_per_frame = defaultdict(list)
    for bbox in inference_bboxes:
        inf_bboxes_per_frame[bbox.frame_id].append(bbox)

    frame_metrics = {}
    for frame_id in gt_bbox_per_frame:
        if frame_id in gt_bbox_per_frame and frame_id in inf_bboxes_per_frame:
            matches = match_inferences(
                [gt_bbox_per_frame[frame_id]],
                inf_bboxes_per_frame[frame_id],
            )
            frame_metrics[frame_id] = FrameMatch(
                frame_id=frame_id,
                matched_pedestrian=matches.matched[0][1] if len(matches.matched) > 0 else None,
                gt=matches.matched[0][0] if len(matches.matched) > 0 else None,
                gt_label=gt_bbox_per_frame[frame_id].label,
                inf_label=matches.matched[0][1].label if len(matches.matched) > 0 else None,
                unmatched_gt=matches.unmatched_gt,
                unmatched_inf=matches.unmatched_inf,
                matched=matches.matched,
            )
    return frame_metrics


def compute_frame_metrics_row(
    gt_bboxes: List[PedestrianBoundingBox],
    high_risk_pids: List[str],
    inference_bboxes: List[ScoredPedestrianBoundingBox],
) -> Tuple[Dict[str, Any], List[ScoredPedestrianBoundingBox]]:
    frame_metrics_combined = {}
    df_raw_inferences = []
    for pid in high_risk_pids:
        filtered_gt_bboxes = [bbox for bbox in gt_bboxes if bbox.ped_id == pid]
        filtered_inference_bboxes = [
            bbox for bbox in inference_bboxes if bbox.ped_id == pid and bbox.score >= CONFIDENCE
        ]
        df_raw_inferences.extend(filtered_inference_bboxes)
        frame_metrics_combined[pid] = compute_pedestrian_metrics(filtered_gt_bboxes, filtered_inference_bboxes)

    return frame_metrics_combined, df_raw_inferences


def compute_match_arrays(matches: List[FrameMatch]) -> FrameMetrics:
    tps, fps, tns, fns = [], [], [], []
    for match in matches:
        if match.matched_pedestrian is not None:
            if match.inf_label == "is_crossing":
                if match.gt_label == match.inf_label and match.matched_pedestrian.score >= THRESHOLD:
                    tps.append(match.matched_pedestrian)
                elif match.gt_label != match.inf_label and match.matched_pedestrian.score >= THRESHOLD:
                    continue
            else:
                if match.gt_label != match.inf_label and match.matched_pedestrian.score < THRESHOLD:
                    continue
                else:
                    tns.append(match.gt)

        fps.extend([inf for inf in match.unmatched_inf if inf.score >= THRESHOLD])
        fns.extend(match.unmatched_gt + [gt for gt, inf in match.matched if inf.score < THRESHOLD])

    return FrameMetrics(TP=tps, FP=fps, TN=tns, FN=fns)  # type: ignore


def compute_metrics(dataset: str, inference_data: Dict[str, List[ScoredPedestrianBoundingBox]]) -> pd.DataFrame:
    dataset_df = download_dataset(dataset)
    results = []
    for row in dataset_df.itertuples():
        filemapping = Path(row.locator.split("/")[-1]).stem
        inference_bboxes = inference_data[filemapping]
        frame_metrics_combined, df_raw_inferences = compute_frame_metrics_row(
            row.high_risk,
            row.high_risk_pids,
            inference_bboxes,
        )
        all_pedestrian_by_frames = defaultdict(list)
        for pid, frame_metrics in frame_metrics_combined.items():
            for frame_id, metrics in frame_metrics.items():
                all_pedestrian_by_frames[frame_id].append(metrics)

        cross_frame_precisions, cross_frame_recalls, cross_frame_f1 = [], [], []
        final_tps, final_fps, final_fns, final_tns = [], [], [], []
        frame_metrics_lst = []
        cross_frame_tp_rate = []

        for frame_id, match_lst in all_pedestrian_by_frames.items():
            frame_metrics_match = compute_match_arrays(match_lst)
            tp_count = len(frame_metrics_match.TP)
            fp_count = len(frame_metrics_match.FP)
            fns_count = len(frame_metrics_match.FN)
            final_tps.extend(frame_metrics_match.TP)
            final_fps.extend(frame_metrics_match.FP)
            final_fns.extend(frame_metrics_match.FN)
            final_tns.extend(frame_metrics_match.TN)

            frame_precision = precision(tp_count, fp_count)
            frame_recall = recall(tp_count, fns_count)
            frame_f1 = f1_score(tp_count, fp_count, fns_count)
            cross_frame_precisions.append(frame_precision)
            cross_frame_recalls.append(frame_recall)
            cross_frame_f1.append(frame_f1)
            cross_frame_tp_rate.append(tp_count / len(row.high_risk))
            frame_metrics_lst.append(
                {
                    "frame_id": frame_id,
                    "FrameLevelPrecision": frame_precision,
                    "FrameLevelRecall": frame_recall,
                    "FrameLevelF1": frame_f1,
                },
            )

        results.append(
            {
                "raw_inferences": df_raw_inferences,
                "frame_metrics": frame_metrics_lst,
                "TP": final_tps,
                "FP": final_fps,
                "FN": final_fns,
                "TN": final_tns,
                "has_TP": True if len(final_tps) > 0 else False,
                "has_FP": True if len(final_fps) > 0 else False,
                "has_FN": True if len(final_fns) > 0 else False,
                "has_TN": True if len(final_tns) > 0 else False,
                "CrossFramePrecision": np.mean(cross_frame_precisions),
                "CrossFrameRecall": np.mean(cross_frame_recalls),
                "CrossFrameF1": np.mean(cross_frame_f1),
                "CrossFrameTPR": np.mean(cross_frame_tp_rate),
                "locator": row.locator,
            },
        )
    return pd.DataFrame(results)


def run(args: Namespace) -> None:
    eval_config = dict(
        iou_threshold=THRESHOLD,
        threshold_strategy=THRESHOLD,
        min_confidence_score=CONFIDENCE,
    )
    split_model = args.model.split("_")
    inference_data = process_inf_data(split_model[0], split_model[1])
    results_df = compute_metrics(args.dataset, inference_data)
    upload_results(args.dataset, args.model, [(eval_config, results_df)])


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
