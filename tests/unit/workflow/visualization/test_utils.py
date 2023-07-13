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
import os

import numpy as np
import pytest
from PIL import Image

from kolena.errors import InputValidationError
from kolena.workflow.visualization import encode_png


@pytest.mark.parametrize(
    "image",
    [
        np.array([[]], dtype=int),
        np.array([], dtype=np.uint8),
        np.array([[]], dtype=float),
        np.array([[]], dtype=np.uint8),
        np.array([[[]]], dtype=np.uint8),
        np.array([[[[]]]], dtype=np.uint8),
        np.array([1, 2, 3], dtype=np.uint8),
        np.array([[[[1], [2], [3]]]], dtype=np.uint8),
    ],
)
def test__encode_png__invalid_input(image: np.ndarray) -> None:
    with pytest.raises(InputValidationError):
        encode_png(image, mode="RGB")


def load_test_image(mode: str) -> np.ndarray:
    image = Image.open(f"{os.path.dirname(__file__)}/data/rgba.png").convert(mode)
    return np.asarray(image)


@pytest.mark.parametrize(
    "mode",
    [
        "RGB",
        "RGBA",
        "L",
    ],
)
def test__encode_png(mode: str) -> None:
    # testing for a complete run without any errors raised
    image = load_test_image(mode)
    encode_png(image, mode)


def test__encode_png__invalid_mode() -> None:
    image = load_test_image("RGB")

    with pytest.raises(ValueError):
        encode_png(image, "INVALID")
