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
import tempfile
from io import BytesIO
from typing import List
from typing import Tuple
from urllib.parse import urlparse

import boto3
import cv2
import numpy as np
import skimage

from kolena.workflow.visualization import colorize_activation_map
from kolena.workflow.visualization import encode_png

s3 = boto3.client("s3")


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


def sanitize_model_name(model_name: str) -> str:
    """
    See https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-keys.html for object-key name recommendations
    Safe characters are alphanumeric or a set of some special characters.
    """
    special_chars = {"!", "-", "_", ".", "*", "(", ")"}
    output_string = ""
    for character in model_name:
        if character.isalnum() or character in special_chars:
            output_string += character
    return output_string


def compute_sklearn_arrays(gt_masks: List[np.ndarray], inf_probs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.concatenate([gt_mask.ravel() for gt_mask in gt_masks])
    y_pred = np.concatenate([inf_prob.ravel() for inf_prob in inf_probs])
    y_true[y_true != 1] = 0  # binarize gt_mask
    return y_true, y_pred


def compute_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> Tuple[float, float, float]:
    tp = np.sum(np.logical_and(y_true == 1, y_pred >= threshold))
    fp = np.sum(np.logical_and(y_true == 0, y_pred >= threshold))
    fn = np.sum(np.logical_and(y_true == 1, y_pred < threshold))

    precision = tp / float(tp + fp) if tp + fp > 0 else 0.0
    recall = tp / float(tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0

    return precision, recall, f1


def create_bitmap(activation_map: np.ndarray) -> BytesIO:
    bitmap = colorize_activation_map(activation_map)
    image_buffer = encode_png(bitmap, mode="RGBA")
    return image_buffer
