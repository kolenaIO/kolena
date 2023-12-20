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
import cv2
import numpy as np
import s3fs


def download_image(locator: str) -> np.ndarray:
    s3 = s3fs.S3FileSystem(anon=True)
    with s3.open(locator, "rb") as f:
        image_arr = np.asarray(bytearray(f.read()), dtype="uint8")
        image = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
        return image
