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
import tempfile
from io import BytesIO
from typing import Tuple
from urllib.parse import urlparse

import boto3
import cv2
import numpy as np
import skimage

from kolena.workflow.visualization import colorize_activation_map
from kolena.workflow.visualization import encode_png

s3 = boto3.client("s3")


def load_data(gt_locator: str, inf_locator: str) -> Tuple[np.ndarray, np.ndarray]:
    inf_proba = download_binary_array(inf_locator)
    gt_mask = download_mask(gt_locator)
    gt_mask[gt_mask != 1] = 0  # binarize gt_mask
    return gt_mask, inf_proba


def apply_threshold(proba_mask: np.ndarray, threshold: float) -> np.ndarray:
    mask = np.zeros_like(proba_mask)
    mask[proba_mask >= threshold] = 1
    return mask


def compute_result_masks(gt_mask: np.ndarray, inf_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp_mask = np.where(gt_mask != inf_mask, 0, inf_mask)
    fp_mask = np.where(gt_mask == inf_mask, 0, inf_mask)
    fn_mask = np.where(gt_mask == inf_mask, 0, gt_mask)

    return tp_mask, fp_mask, fn_mask


def upload_result_masks(
    tp_mask: np.ndarray,
    fp_mask: np.ndarray,
    fn_mask: np.ndarray,
    locator_prefix: str,
    basename: str,
) -> Tuple[str, str, str]:
    tp_locator = f"{locator_prefix}/TP/{basename}.png"
    fp_locator = f"{locator_prefix}/FP/{basename}.png"
    fn_locator = f"{locator_prefix}/FN/{basename}.png"

    upload_image(tp_locator, tp_mask)
    upload_image(fp_locator, fp_mask)
    upload_image(fn_locator, fn_mask)

    return tp_locator, fp_locator, fn_locator


def parse_s3_path(s3_path: str) -> Tuple[str, str]:
    """
    Parse an s3 path of the form s3://bucket/key
    """
    parsed_url = urlparse(str(s3_path))
    s3_bucket = parsed_url.netloc
    s3_key = parsed_url.path.lstrip("/")
    return s3_bucket, s3_key


def download_mask(locator: str) -> np.ndarray:
    """
    Download a mask stored on s3 and return it as a np array
    """
    bucket, key = parse_s3_path(locator)
    with tempfile.TemporaryFile() as f:
        s3.download_fileobj(bucket, key, f)
        data = skimage.io.imread(f)
        return data


def download_binary_array(locator: str) -> np.ndarray:
    """
    Download a .npy stored on s3 and return it as a np array
    NOTE: couldn't use `download_fileobj` as it was raising 'ValueError: cannot reshape array of size ... into
    shape (..., ...)' — seems like the download was incomplete
    """
    bucket, key = parse_s3_path(locator)
    obj_response = s3.get_object(Bucket=bucket, Key=key)
    with BytesIO(obj_response["Body"].read()) as f:
        f.seek(0)
        return np.load(f)


def upload_image_buffer(locator: str, io_buf: BytesIO) -> None:
    bucket, key = parse_s3_path(locator)
    s3.upload_fileobj(io_buf, bucket, key)


def upload_image(locator: str, image: np.ndarray) -> None:
    success, buf = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("failed to encode image as PNG")

    io_buf = BytesIO(buf)
    upload_image_buffer(locator, io_buf)


def create_bitmap(activation_map: np.ndarray) -> BytesIO:
    bitmap = colorize_activation_map(activation_map)
    image_buffer = encode_png(bitmap, mode="RGBA")
    return image_buffer
