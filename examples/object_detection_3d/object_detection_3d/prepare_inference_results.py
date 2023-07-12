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

import numpy as np
import open3d
import pandas as pd
from mmdet3d.apis import inference_detector
from mmdet3d.apis import init_model
from mmdet3d.evaluation import KittiMetric
from mmdet3d.structures import Box3DMode
from object_detection_3d.utils import calibration_from_label_id
from object_detection_3d.utils import get_label_path
from object_detection_3d.utils import get_lidar_label_path
from object_detection_3d.utils import get_result_path
from object_detection_3d.utils import get_velodyne_path
from object_detection_3d.utils import get_velodyne_pcd_path
from object_detection_3d.utils import LABEL_FILE_COLUMNS
from tqdm import tqdm


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
        json.dump(dict(results=results, classes=classes), f, indent=2)


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


def convert_kitti_to_pcd(lidar_file: str, target_file: str) -> None:
    bin_pcd = np.fromfile(lidar_file, dtype=np.float32)
    points = bin_pcd.reshape((-1, 4))[:, 0:3]
    o3d_pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points))
    open3d.io.write_point_cloud(target_file, o3d_pcd, compressed=True)


def calibrate_velo_to_cam(datadir: Path) -> None:
    """Convert KITTI 3D bbox coordinates to image coordinates"""

    lidar_label_dir = get_lidar_label_path(datadir)
    if not os.path.exists(lidar_label_dir):
        os.mkdir(lidar_label_dir)

    for label_filepath in tqdm(glob.glob(f"{get_label_path(datadir)}/*.txt")):
        label_id = Path(label_filepath).stem
        lidar_label_filepath = get_lidar_label_path(datadir) / f"{label_id}.txt"
        if not os.path.exists(lidar_label_filepath):
            continue

        df = pd.read_csv(label_filepath, delimiter=" ", header=None, names=LABEL_FILE_COLUMNS)
        calibration = calibration_from_label_id(datadir, label_id)

        camera_bboxes = [
            [record.loc_x, record.loc_y, record.loc_z, record.dim_x, record.dim_y, record.dim_z, record.rotation_y]
            for record in df.itertuples()
        ]

        lidar_bboxes = Box3DMode.convert(
            np.array(camera_bboxes),
            Box3DMode.CAM,
            Box3DMode.LIDAR,
            rt_mat=np.linalg.inv(calibration["camera_rectification"] @ calibration["velodyne_to_camera"]),
            with_yaw=True,
        )

        df["loc_x"] = [bbox[0] for bbox in lidar_bboxes]
        df["loc_y"] = [bbox[1] for bbox in lidar_bboxes]
        df["loc_z"] = [bbox[2] for bbox in lidar_bboxes]
        df["dim_x"] = [bbox[3] for bbox in lidar_bboxes]
        df["dim_y"] = [bbox[4] for bbox in lidar_bboxes]
        df["dim_z"] = [bbox[5] for bbox in lidar_bboxes]
        df["rotation_y"] = [bbox[6] for bbox in lidar_bboxes]

        df.to_csv(lidar_label_filepath, header=False, index=False, sep=" ")


def prepare_pcd(datadir: Path) -> None:
    """Convert KITTI velodyne bin format to PCD for visualization"""

    velodyne_path = get_velodyne_path(datadir)
    pcd_path = get_velodyne_pcd_path(datadir)
    if not os.path.exists(pcd_path):
        os.mkdir(pcd_path)

    for bin in glob.glob(f"{velodyne_path}/*.bin"):
        filename = os.path.basename(bin)
        pcd = str(pcd_path / f"{os.path.splitext(filename)[0]}.pcd")
        if not os.path.exists(pcd):
            convert_kitti_to_pcd(bin, pcd)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("datadir", help="Data dir", type=Path)
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
    calibrate_velo_to_cam(args.datadir)
    prepare_pcd(args.datadir)
    prepare_inferences(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
