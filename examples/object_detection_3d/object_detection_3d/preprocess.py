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
import glob
import pickle
import struct
from pathlib import Path
from typing import Any
from typing import Dict

import numpy as np
import open3d
import pandas as pd
import torch
from mmdet3d.structures import Box3DMode
from mmdet3d.structures import points_cam2img

stem = "hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.pkl"
mmdet_results_file = "/data/open-source/kitti/3d-object-detection/results/mmdet_format/" + stem
results_file = "/data/open-source/kitti/3d-object-detection/results/" + stem
mmdet_results = pickle.load(open(mmdet_results_file, "rb"))

annotation_file = "mmdetection3d/data/kitti/kitti_infos_trainval.pkl"
annotations = pickle.load(open(annotation_file, "rb"))

pcd_limit_range = [0, -40, -3, 70.4, 40, 0.0]


def convert_valid_bboxes(box_dict: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
    """Convert the predicted boxes into valid ones.
    Args:
        box_dict (dict): Box dictionaries to be converted.
            - boxes_3d (:obj:`LiDARInstance3DBoxes`): 3D bounding boxes.
            - scores_3d (torch.Tensor): Scores of boxes.
            - labels_3d (torch.Tensor): Class labels of boxes.
        info (dict): Data info.
    Returns:
        dict: Valid predicted boxes.
            - bbox (np.ndarray): 2D bounding boxes.
            - box3d_camera (np.ndarray): 3D bounding boxes in
                camera coordinate.
            - box3d_lidar (np.ndarray): 3D bounding boxes in
                LiDAR coordinate.
            - scores (np.ndarray): Scores of boxes.
            - label_preds (np.ndarray): Class label predictions.
            - sample_idx (int): Sample index.
    """
    # TODO: refactor this function
    box_preds = box_dict["boxes_3d"]
    scores = box_dict["scores_3d"]
    labels = box_dict["labels_3d"]
    sample_idx = info["image"]["image_idx"]
    box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

    if len(box_preds) == 0:
        return dict(
            bbox=np.zeros([0, 4]),
            box3d_camera=np.zeros([0, 7]),
            box3d_lidar=np.zeros([0, 7]),
            scores=np.zeros([0]),
            label_preds=np.zeros([0, 4]),
            sample_idx=sample_idx,
        )

    rect = info["calib"]["R0_rect"].astype(np.float32)
    Trv2c = info["calib"]["Tr_velo_to_cam"].astype(np.float32)
    P2 = info["calib"]["P2"].astype(np.float32)
    img_shape = info["image"]["image_shape"]
    P2 = box_preds.tensor.new_tensor(P2)

    box_preds_camera = box_preds.convert_to(Box3DMode.CAM, rect @ Trv2c)

    box_corners = box_preds_camera.corners
    box_corners_in_image = points_cam2img(box_corners, P2)
    # box_corners_in_image: [N, 8, 2]
    minxy = torch.min(box_corners_in_image, dim=1)[0]
    maxxy = torch.max(box_corners_in_image, dim=1)[0]
    box_2d_preds = torch.cat([minxy, maxxy], dim=1)
    # Post-processing
    # check box_preds_camera
    image_shape = box_preds.tensor.new_tensor(img_shape)
    valid_cam_inds = (
        (box_2d_preds[:, 0] < image_shape[1])
        & (box_2d_preds[:, 1] < image_shape[0])
        & (box_2d_preds[:, 2] > 0)
        & (box_2d_preds[:, 3] > 0)
    )
    # check box_preds
    limit_range = box_preds.tensor.new_tensor(pcd_limit_range)
    valid_pcd_inds = (box_preds.center > limit_range[:3]) & (box_preds.center < limit_range[3:])
    valid_inds = valid_cam_inds & valid_pcd_inds.all(-1)

    if valid_inds.sum() > 0:
        return dict(
            bbox=box_2d_preds[valid_inds, :].numpy(),
            box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
            box3d_lidar=box_preds[valid_inds].tensor.numpy(),
            scores=scores[valid_inds].numpy(),
            label_preds=labels[valid_inds].numpy(),
            sample_idx=sample_idx,
        )
    else:
        return dict(
            bbox=np.zeros([0, 4]),
            box3d_camera=np.zeros([0, 7]),
            box3d_lidar=np.zeros([0, 7]),
            scores=np.zeros([0]),
            label_preds=np.zeros([0, 4]),
            sample_idx=sample_idx,
        )


results = []
for result, ann in zip(mmdet_results, annotations):
    bboxes = convert_valid_bboxes(result, ann)
    results.append(
        {
            "image_id": ann["image"]["image_idx"],
            "image_path": ann["image"]["image_path"],
            "velodyne_path": ann["point_cloud"]["velodyne_path"],
            "calib": ann["calib"],
            "bbox": bboxes["bbox"],
            "box3d_camera": bboxes["box3d_camera"],
            "box3d_lidar": bboxes["box3d_lidar"],
            "scores": bboxes["scores"],
            "label_preds": bboxes["label_preds"],
        },
    )

sorted_result = sorted(results, key=lambda d: d["image_id"])
pickle.dump(sorted_result, open(results_file, "wb"))

DATA_DIR = "/data/open-source/kitti/3d-object-detection/training/"
LABEL_FILE_COLUMNS = [
    "type",
    "truncated",
    "occluded",
    "alpha",
    "bbox_x0",
    "bbox_y0",
    "bbox_x1",
    "bbox_y1",
    "dim_y",
    "dim_z",
    "dim_x",
    "loc_x",
    "loc_y",
    "loc_z",
    "rotation_y",
]


def gt_from_label_id(label_id: str) -> pd.DataFrame:
    label_filepath = Path(DATA_DIR) / "label_2" / f"{label_id}.txt"
    df = pd.read_csv(label_filepath, delimiter=" ", header=None, names=LABEL_FILE_COLUMNS)
    return df


def calibration_from_label_id(label_id: str) -> Dict[str, np.ndarray]:
    def extend_matrix(mat: np.ndarray) -> np.ndarray:
        return np.concatenate([mat, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)

    calibration_filepath = Path(DATA_DIR) / "calib" / f"{label_id}.txt"
    with open(calibration_filepath) as f:
        lines = f.readlines()

    # p0 = extend_matrix(np.array([float(info) for info in lines[0].split(" ")[1:13]]).reshape([3, 4]))
    # p1 = extend_matrix(np.array([float(info) for info in lines[1].split(" ")[1:13]]).reshape([3, 4]))
    p2 = extend_matrix(np.array([float(info) for info in lines[2].split(" ")[1:13]]).reshape([3, 4]))
    # p3 = extend_matrix(np.array([float(info) for info in lines[3].split(" ")[1:13]]).reshape([3, 4]))
    r0_rect = np.array([float(info) for info in lines[4].split(" ")[1:10]]).reshape([3, 3])
    rect_4x4 = np.zeros([4, 4], dtype=r0_rect.dtype)
    rect_4x4[3, 3] = 1.0
    rect_4x4[:3, :3] = r0_rect
    velo_to_cam = extend_matrix(np.array([float(info) for info in lines[5].split(" ")[1:13]]).reshape([3, 4]))

    return dict(
        velodyne_to_camera=velo_to_cam,
        camera_rectification=rect_4x4,
        camera_to_image=p2,
    )


label_files = glob.glob(str(Path(DATA_DIR) / "label_2") + "/*.txt")
label_ids = [Path(filepath).stem for filepath in label_files]

for label_id in label_ids:
    df = gt_from_label_id(label_id)
    calibration = calibration_from_label_id(label_id)

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

    df.to_csv(Path(DATA_DIR) / "label_2_lidar" / f"{label_id}.txt", header=False, index=False, sep=" ")


def convert_kitti_bin_to_pcd(binFilePath: str) -> open3d.geometry.PointCloud:
    size_float = 4
    list_pcd = []
    with open(binFilePath, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np_pcd)
    return pcd
