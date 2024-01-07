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
from abc import ABC
from abc import abstractmethod

import numpy as np

from kolena.errors import InputValidationError


class Colormap(ABC):
    """
    A `Colormap` maps a pixel intensity to RGBA.
    """

    fade_low_activation: bool
    """
    Fades out the regions with low activation by applying zero alpha value if set
    `True`; otherwise, activation map is shown as is without any fading applied. By default, it's set to `True`.
    This option makes the overlay visualization better by highlighting only the important regions.
    """

    def __init__(self, fade_low_activation: bool = True):
        self.fade_low_activation = fade_low_activation

    @abstractmethod
    def red(self, intensity: np.uint8) -> np.uint8:
        """
        Maps a grayscale pixel intensity to color red: [0, 255]
        """
        raise NotImplementedError

    @abstractmethod
    def green(self, intensity: np.uint8) -> np.uint8:
        """
        Maps a grayscale pixel intensity to color green: [0, 255]
        """
        raise NotImplementedError

    @abstractmethod
    def blue(self, intensity: np.uint8) -> np.uint8:
        """
        Maps a grayscale pixel intensity to color blue: [0, 255]
        """
        raise NotImplementedError

    def alpha(self, intensity: np.uint8) -> np.uint8:
        """
        Maps the grayscale pixel intensity to alpha: [0, 255]. If `fade_low_activation`
        is False, then it returns the maximum alpha value.
        """
        max_uint8 = np.iinfo(np.uint8).max
        if self.fade_low_activation:
            # apply non-linear scaling alpha channel [0, 255]
            return max_uint8 / (1 + np.exp(((max_uint8 / 2) - intensity) / 8))
        return np.uint8(max_uint8)

    def colorize(self, intensity: np.uint8) -> np.array:
        return np.array(
            [
                self.red(intensity),
                self.green(intensity),
                self.blue(intensity),
                self.alpha(intensity),
            ],
            dtype=np.uint8,
        )

    def _interpolate(self, intensity: np.uint8, x0: float, y0: float, x1: float, y1: float) -> np.uint8:
        return np.uint8(round((intensity - x0) * (y1 - y0) / (x1 - x0) + y0))


class ColormapJet(Colormap):
    """
    The [MATLAB "Jet" color palette](http://blogs.mathworks.com/images/loren/73/colormapManip_14.png) is a standard
    palette used for scientific and mathematical data.

    It is defined as a linear ramp between the following colours: "#00007F", "blue", "#007FFF", "cyan", "#7FFF7F",
    "yellow", "#FF7F00", "red", "#7F0000"
    """

    def _scale_to_colormap(self, intensity: np.uint8) -> np.uint8:
        max_uint8 = np.iinfo(np.uint8).max
        if intensity <= max_uint8 / 8:
            return np.uint8(0)
        elif intensity <= max_uint8 * 3 / 8:
            return self._interpolate(intensity, max_uint8 / 8, 0.0, max_uint8 * 3 / 8, max_uint8)
        elif intensity <= max_uint8 * 5 / 8:
            return np.uint8(max_uint8)
        elif intensity <= max_uint8 * 7 / 8:
            return self._interpolate(intensity, max_uint8 * 5 / 8, max_uint8, max_uint8 * 7 / 8, 0.0)
        else:
            return np.uint8(0)

    def red(self, intensity: np.uint8) -> np.uint8:
        return self._scale_to_colormap(intensity - np.iinfo(np.uint8).max / 4)  # type: ignore

    def green(self, intensity: np.uint8) -> np.uint8:
        return self._scale_to_colormap(intensity)

    def blue(self, intensity: np.uint8) -> np.uint8:
        return self._scale_to_colormap(intensity + np.iinfo(np.uint8).max / 4)  # type: ignore


def colorize_activation_map(activation_map: np.ndarray, colormap: Colormap = ColormapJet()) -> np.ndarray:
    """
    Applies the specified colormap to the activation map.

    :param activation_map: A 2D numpy array, shaped (h, w) or (h, w, 1), of the activation map in `np.uint8`
        or `float` ranging [0, 1].
    :param colormap: The colormap used to colorize the input activation map. Defaults to the
        [MATLAB "Jet" colormap](http://blogs.mathworks.com/images/loren/73/colormapManip_14.png).
    :return: The colorized activation map in RGBA format, in (h, w, 4) shape.
    """
    max_uint8 = np.iinfo(np.uint8).max

    if activation_map.size == 0:
        raise InputValidationError("input array is empty")

    if np.issubdtype(activation_map.dtype, np.floating) and ((0 <= activation_map) & (activation_map <= 1)).all():
        activation_map = np.rint(activation_map * max_uint8).astype(np.uint8)

    if activation_map.dtype != np.uint8:
        raise InputValidationError(f"input array type must be np.uint8, but received {activation_map.dtype}")

    if len(activation_map.shape) != 2 and len(activation_map.shape) != 3:
        raise InputValidationError(
            f"input array must have 2 or 3 dimensions, but received {len(activation_map.shape)}\n"
            "Expected shape is (h, w) or (h, w, 1)",
        )

    if len(activation_map.shape) == 2:
        activation_map = np.expand_dims(activation_map, axis=2)

    if len(activation_map.shape) == 3:
        if activation_map.shape[2] != 1:
            raise InputValidationError(
                f"input image must be a single-channel, but received {activation_map.shape[2]} channels\n"
                "Expected shape is (h, w) or (h, w, 1)",
            )
        activation_map = np.repeat(activation_map, 4, axis=2)

    vcolorize = np.vectorize(colormap.colorize, signature="()->(n)")
    activation_map[:, :] = vcolorize(activation_map[:, :, 0])

    return activation_map
