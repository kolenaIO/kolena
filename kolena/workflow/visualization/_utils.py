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
import io

import numpy as np
from PIL import Image

from kolena.errors import InputValidationError


def encode_png(image: np.ndarray, mode: str) -> io.BytesIO:
    """
    Encodes an image into an in-memory PNG file that is represented as binary data. It is used when you want to upload
    a 2 or 3-dimensional image in a NumPy array format to cloud.

    It can be used in conjunction with
    [`colorized_activation_map`][kolena.workflow.visualization.colorize_activation_map] when uploading an
    activation map.

    :param image: A 2D or 3D NumPy array, shaped either `(h, w)`, `(h, w, 1)`, `(h, w, 3)`, or `(h, w, 4)`
    :param mode: A [PIL mode](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes)
    :return: The in-memory PNG file represented as binary data.
    """
    if image.size == 0:
        raise InputValidationError("input array is empty")

    if len(image.shape) < 2 or len(image.shape) > 3:
        raise InputValidationError(
            f"input array must have 2 or 3 dimensions, but received {len(image.shape)}\n"
            "Here are supported input shape: (h, w), (h, w, 1), (h, w, 3), or (h, w, 4)",
        )

    pil_image = Image.fromarray(image, mode=mode)
    image_buf = io.BytesIO()
    pil_image.save(image_buf, format="png")
    image_buf.seek(0)

    return image_buf
