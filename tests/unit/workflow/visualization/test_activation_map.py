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
import numpy as np
import pytest

from kolena.errors import InputValidationError
from kolena.workflow.visualization import colorize_activation_map
from kolena.workflow.visualization import ColormapJet


@pytest.mark.parametrize(
    "activation_map",
    [
        np.array([[]], dtype=int),
        np.array([], dtype=np.uint8),
        np.array([[]], dtype=float),
        np.array([[0.4], [-0.4], [3.0]], dtype=float),
        np.array([[0.4], [-0.4], [3.0]], dtype=np.float64),
        np.array([[]], dtype=np.uint8),
        np.array([[[]]], dtype=np.uint8),
        np.array([[[[]]]], dtype=np.uint8),
        np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.uint8),
    ],
)
def test__colorize_activation_map__invalid_input(activation_map: np.ndarray) -> None:
    with pytest.raises(InputValidationError):
        colorize_activation_map(activation_map)

    with pytest.raises(InputValidationError):
        colorize_activation_map(activation_map, colormap=ColormapJet(fade_low_activation=False))


@pytest.mark.parametrize(
    "activation_map, fade_low_activation, expected",
    [
        (
            np.array([[0, 0, 0], [0, 0, 0]], dtype=np.uint8),
            True,
            np.array(
                [
                    [
                        [0, 0, 128, 0],
                        [0, 0, 128, 0],
                        [0, 0, 128, 0],
                    ],
                    [
                        [0, 0, 128, 0],
                        [0, 0, 128, 0],
                        [0, 0, 128, 0],
                    ],
                ],
            ),
        ),
        (
            np.array([[0, 0, 0], [0, 0, 0]], dtype=np.uint8),
            False,
            np.array(
                [
                    [
                        [0, 0, 128, 255],
                        [0, 0, 128, 255],
                        [0, 0, 128, 255],
                    ],
                    [
                        [0, 0, 128, 255],
                        [0, 0, 128, 255],
                        [0, 0, 128, 255],
                    ],
                ],
            ),
        ),
        (
            np.array([[32, 95, 128]], dtype=np.uint8),
            False,
            np.array(
                [
                    [
                        [0, 0, 255, 255],
                        [0, 252, 255, 255],
                        [130, 255, 126, 255],
                    ],
                ],
            ),
        ),
        (
            np.array([[160, 223, 255]], dtype=np.uint8),
            False,
            np.array(
                [
                    [
                        [255, 252, 0, 255],
                        [255, 0, 0, 255],
                        [128, 0, 0, 255],
                    ],
                ],
            ),
        ),
        (
            np.array([[32, 95, 128]], dtype=np.uint8),
            True,
            np.array(
                [
                    [
                        [0, 0, 255, 0],
                        [0, 252, 255, 4],
                        [130, 255, 126, 131],
                    ],
                ],
            ),
        ),
        (
            np.array([[160, 223, 255]], dtype=np.uint8),
            True,
            np.array(
                [
                    [
                        [255, 252, 0, 250],
                        [255, 0, 0, 254],
                        [128, 0, 0, 254],
                    ],
                ],
            ),
        ),
        (
            np.array([[[0], [0], [0]], [[0], [0], [0]]], dtype=np.uint8),
            False,
            np.array(
                [
                    [
                        [0, 0, 128, 255],
                        [0, 0, 128, 255],
                        [0, 0, 128, 255],
                    ],
                    [
                        [0, 0, 128, 255],
                        [0, 0, 128, 255],
                        [0, 0, 128, 255],
                    ],
                ],
            ),
        ),
        (
            np.array([[0.627, 0.875, 1.0]], dtype=float),
            True,
            np.array(
                [
                    [
                        [255, 252, 0, 250],
                        [255, 0, 0, 254],
                        [128, 0, 0, 254],
                    ],
                ],
            ),
        ),
        (
            np.array([[0.627, 0.875, 1.0]], dtype=np.float32),
            True,
            np.array(
                [
                    [
                        [255, 252, 0, 250],
                        [255, 0, 0, 254],
                        [128, 0, 0, 254],
                    ],
                ],
            ),
        ),
        (
            np.array([[0.627, 0.875, 1.0]], dtype=np.float64),
            True,
            np.array(
                [
                    [
                        [255, 252, 0, 250],
                        [255, 0, 0, 254],
                        [128, 0, 0, 254],
                    ],
                ],
            ),
        ),
    ],
)
def test__colorize_activation_map(
    activation_map: np.ndarray,
    fade_low_activation: bool,
    expected: np.ndarray,
) -> None:
    colorized_map = colorize_activation_map(
        activation_map,
        colormap=ColormapJet(fade_low_activation=fade_low_activation),
    )
    np.testing.assert_array_equal(expected, colorized_map)


@pytest.mark.parametrize(
    "intensity, fade_low_activation, expected",
    [
        (0, True, [0, 0, 128, 0]),
        (0, False, [0, 0, 128, 255]),
        (255, True, [128, 0, 0, 254]),
        (255, False, [128, 0, 0, 255]),
        (128, True, [130, 255, 126, 131]),
        (128, False, [130, 255, 126, 255]),
        (32, True, [0, 0, 255, 0]),
        (32, False, [0, 0, 255, 255]),
        (95, True, [0, 252, 255, 4]),
        (95, False, [0, 252, 255, 255]),
        (160, True, [255, 252, 0, 250]),
        (160, False, [255, 252, 0, 255]),
        (223, True, [255, 0, 0, 254]),
        (223, False, [255, 0, 0, 255]),
    ],
)
def test__colormap_jet(
    intensity: np.uint8,
    fade_low_activation: bool,
    expected: np.array,
) -> None:
    colormap = ColormapJet(fade_low_activation=fade_low_activation)
    np.testing.assert_array_equal(expected, colormap.colorize(intensity))
