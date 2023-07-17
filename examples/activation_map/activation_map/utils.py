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
import io
import os
from typing import Tuple
from urllib.parse import urlparse

import boto3
import numpy as np

from kolena.workflow.annotation import BitmapMask
from kolena.workflow.visualization import colorize_activation_map
from kolena.workflow.visualization import encode_png

s3 = boto3.client("s3")


def parse_s3_url(locator: str) -> Tuple[str, str]:
    parsed_url = urlparse(locator)
    s3_bucket = parsed_url.netloc
    s3_key = parsed_url.path.lstrip("/")
    return s3_bucket, s3_key


def load_activation_map(activation_map_locator: str) -> np.ndarray:
    s3_bucket, s3_key = parse_s3_url(activation_map_locator)

    with io.BytesIO(s3.get_object(Bucket=s3_bucket, Key=s3_key)["Body"].read()) as f:
        f.seek(0)
        return np.load(f, allow_pickle=True)


def create_bitmap(activation_map: np.ndarray) -> io.BytesIO:
    bitmap = colorize_activation_map(activation_map)
    image_buffer = encode_png(bitmap, mode="RGBA")
    return image_buffer


def upload_bitmap(image_buffer: io.BytesIO, bitmap_locator: str) -> None:
    s3_bucket, s3_key = parse_s3_url(bitmap_locator)
    s3.upload_fileobj(image_buffer, s3_bucket, s3_key)


def create_and_upload_bitmap(
    bitmap_locator: str,
    activation_map: np.ndarray,
) -> BitmapMask:
    image_buffer = create_bitmap(activation_map)
    upload_bitmap(image_buffer, bitmap_locator)
    return BitmapMask(bitmap_locator)


def bitmap_locator(upload_s3_bucket: str, upload_path: str, filename: str) -> str:
    return os.path.join("s3://", upload_s3_bucket, upload_path, f"{filename}.png")
