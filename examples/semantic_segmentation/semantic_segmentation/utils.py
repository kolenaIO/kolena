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
import botocore
import cv2
import numpy as np
import skimage

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
    with tempfile.NamedTemporaryFile() as f:
        s3.download_fileobj(bucket, key, f)
        data = skimage.io.imread(f)
        return data


def download_binary_array(locator: str) -> np.ndarray:
    bucket, key = parse_s3_path(locator)
    with tempfile.NamedTemporaryFile() as f:
        try:
            s3.download_fileobj(bucket, key, f)
        except botocore.exceptions.ClientError:
            raise ValueError(f"Failed to load s3://{bucket}/{key}")
        return np.load(f.name, allow_pickle=True)


def upload_image(locator: str, image: np.ndarray) -> None:
    bucket, key = parse_s3_path(locator)
    success, buf = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("failed to encode image as PNG")

    io_buf = BytesIO(buf)
    s3.upload_fileobj(io_buf, bucket, key)


def resize(mask: np.ndarray, shape: Tuple[int, int], order: int, preserve_range: bool) -> np.ndarray:
    return skimage.transform.resize(
        mask,
        (shape[0], shape[1]),
        order=order,
        mode="constant",
        anti_aliasing=False,
        preserve_range=preserve_range,
    ).astype(np.float32)


def compute_sklearn_arrays(gt_masks: List[np.ndarray], inf_probs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    return np.array(gt_masks).ravel(), np.array(inf_probs).ravel()  # AKA (y_true, y_pred)
