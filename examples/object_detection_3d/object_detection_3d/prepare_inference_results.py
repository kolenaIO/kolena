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
import csv
import glob
import json
import os.path
import pickle
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

from mmdet3d.apis import inference_detector
from mmdet3d.apis import init_model
from mmdet3d.evaluation import KittiMetric
from object_detection_3d.utils import get_result_path
from object_detection_3d.utils import get_velodyne_path


DEFAULT_RESULT_FILE = "results.json"


def load_anno_file(annofile: Path) -> Dict[str, Any]:
    with open(str(annofile), "rb") as f:
        return pickle.load(f)


def zip_bboxes(bboxes: Dict[str, Any], class_code: Dict[int, str]) -> List[Dict[str, Any]]:
    return [
        dict(
            box=[v.item() for v in bboxes["bbox"][i]],
            box3d=[v.item() for v in bboxes["box3d_lidar"][i]],
            score=bboxes["scores"][i].item(),
            pred=class_code[bboxes["label_preds"][i]],
        )
        for i in range(len(bboxes["label_preds"]))
    ]


def save_prediction(result_path: Path, bboxes: List[Dict[str, Any]]) -> None:
    with open(result_path, "w") as f:
        writer = csv.writer(f, delimiter=" ")
        for bbox in bboxes:
            writer.writerow([bbox["pred"], *bbox["box"], *bbox["box3d"], bbox["score"]])


def save_inferences(result_file: Path, results: List[Dict[str, Any]], classes: List[str]) -> None:
    with open(str(result_file), "w") as f:
        json.dump(dict(results=results, classes=classes), f)


def prepare_inferences(args: Namespace) -> None:
    ann_file = load_anno_file(args.datadir / "kitti_infos_trainval.pkl")
    ann_by_id = {data["sample_idx"]: data for data in ann_file["data_list"]}
    model = init_model(args.config, args.checkpoint, device=args.device)
    results = []
    result_path = get_result_path(args.datadir)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    velodyne_dir = get_velodyne_path(args.datadir)
    kitti_metric = KittiMetric("")
    class_to_category = {v: k for k, v in ann_file["metainfo"]["categories"].items()}
    for pcd in list(glob.glob(f"{velodyne_dir}/*.bin")):
        label_id = Path(pcd).stem
        result, data = inference_detector(model, pcd)
        bboxes = zip_bboxes(
            kitti_metric.convert_valid_bboxes(result.pred_instances_3d, ann_by_id[int(label_id)]),
            class_to_category,
        )
        result_file = result_path / f"{label_id}.txt"
        save_prediction(result_file, bboxes)
        results.append(dict(label_id=label_id, bboxes=bboxes))
    save_inferences(args.datadir / args.result_file, results, model.dataset_meta["classes"])


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("datadir", help="KITTI dataset dir", type=Path)
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device used for inference",
    )
    parser.add_argument("--result-file", help="Result file", default=DEFAULT_RESULT_FILE)
    args = parser.parse_args()
    return args


def main(args: Namespace) -> None:
    prepare_inferences(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
