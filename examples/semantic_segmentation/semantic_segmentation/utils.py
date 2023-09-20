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
from typing import Optional
from typing import Tuple
from typing import Union
from urllib.parse import urlparse

import boto3
import cv2
import numpy as np
import skimage
from semantic_segmentation.workflow import Inference
from semantic_segmentation.workflow import TestSampleMetric

from kolena.workflow import AxisConfig
from kolena.workflow import Histogram

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
        s3.download_fileobj(bucket, key, f)
        return np.load(f.name, allow_pickle=True)


def upload_image(locator: str, image: np.ndarray) -> None:
    bucket, key = parse_s3_path(locator)
    success, buf = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("failed to encode image as PNG")

    io_buf = BytesIO(buf)
    s3.upload_fileobj(io_buf, bucket, key)


def compute_score_distribution_plot(
    score: str,
    metrics: List[Union[TestSampleMetric, Inference]],
    binning_info: Optional[Tuple[float, float, float]] = None,  # start, end, num
    logarithmic: bool = False,
) -> Histogram:
    scores = [getattr(m, score) for m in metrics]
    if logarithmic:
        bins = np.logspace(*binning_info, base=2)
    else:
        bins = np.linspace(*binning_info)

    hist, _ = np.histogram(scores, bins=bins)
    return Histogram(
        title=f"Distribution of {score}",
        x_label=f"{score}",
        y_label="Count",
        buckets=list(bins),
        frequency=list(hist),
        x_config=AxisConfig(type="log") if logarithmic else None,
    )
