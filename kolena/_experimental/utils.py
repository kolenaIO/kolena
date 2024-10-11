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
import math
from sys import float_info
from typing import Optional

"""
 The following Python functions for calculating normal and
    chi-square probabilities and critical values were adapted by
    John Walker's Javascript implementation from C implementations
    written by Gary Perlman of Wang Institute, Tyngsboro, MA
    01879.  Both the original C code and this JavaScript edition
    are in the public domain.  */

/*  POZ  --  probability of normal z value
    Adapted from a polynomial approximation in:
            Ibbetson D, Algorithm 209
            Collected Algorithms of the CACM 1963 p. 616
    Note:
            This routine has six digit accuracy, so it is only useful for absolute
            z values <= 6.  For z values > to 6.0, poz() returns 0.0.
"""


def z_score_to_probability(z: float) -> float:
    Z_MAX = 6.0
    if z == 0.0:
        x = 0.0
    else:
        y = 0.5 * abs(z)
        if y > Z_MAX * 0.5:
            x = 1.0
        elif y < 1.0:
            w = y * y
            x = (
                (
                    (
                        (
                            (
                                (
                                    (((0.000124818987 * w - 0.001075204047) * w + 0.005198775019) * w - 0.019198292004)
                                    * w
                                    + 0.059054035642
                                )
                                * w
                                - 0.151968751364
                            )
                            * w
                            + 0.319152932694
                        )
                        * w
                        - 0.531923007300
                    )
                    * w
                    + 0.797884560593
                )
                * y
                * 2.0
            )
        else:
            y -= 2.0
            x = (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        ((-0.000045255659 * y + 0.000152529290) * y - 0.000019538132)
                                                        * y
                                                        - 0.000676904986
                                                    )
                                                    * y
                                                    + 0.001390604284
                                                )
                                                * y
                                                - 0.000794620820
                                            )
                                            * y
                                            - 0.002034254874
                                        )
                                        * y
                                        + 0.006549791214
                                    )
                                    * y
                                    - 0.010557625006
                                )
                                * y
                                + 0.011630447319
                            )
                            * y
                            - 0.009279453341
                        )
                        * y
                        + 0.005353579108
                    )
                    * y
                    - 0.002141268741
                )
                * y
                + 0.000535310849
            ) * y + 0.999936657524
    if z > 0.0:
        return (x + 1.0) * 0.5
    else:
        return (1.0 - x) * 0.5


def probability_to_z_score(p: float) -> float:
    Z_MAX = 6.0
    Z_EPSILON = 0.000001  # Accuracy of z approximation
    minz = -Z_MAX
    maxz = Z_MAX
    zval = 0.0
    pp = p
    if pp < 0.0:
        pp = 0.0
    if pp > 1.0:
        pp = 1.0

    while maxz - minz > Z_EPSILON:
        pval = z_score_to_probability(zval)
        if pval > pp:
            maxz = zval
        else:
            minz = zval
        zval = (maxz + minz) * 0.5
    return zval


def confidence_level_to_z_score(confidence_level: float) -> float:
    if confidence_level < 0 or confidence_level >= 1:
        raise ValueError(f"Invalid confidence level: value '{confidence_level}' must be in [0, 1)")
    probability = (1 + confidence_level) / 2.0
    return probability_to_z_score(probability)


def margin_of_error(n_samples: int, confidence_level: float = 0.95, positive_sample_rate: float = 0.5) -> float:
    if n_samples <= 0:
        return 0.0
    z_score = confidence_level_to_z_score(confidence_level)
    margin_error = z_score * math.sqrt((positive_sample_rate * (1 - positive_sample_rate)) / n_samples)
    return margin_error * 100  # percentage


def round_up_to_nearest_power_of_10(value: float) -> int:
    return 10 ** math.ceil(math.log10(value))


def get_delta_percentage(value_a: float, value_b: float, max_value: Optional[int] = None) -> float:
    upper_limit = None

    if max_value is not None and max_value > 0:
        upper_limit = round_up_to_nearest_power_of_10(max_value)
        if upper_limit >= 1000:
            upper_limit = max_value
        if math.isnan(upper_limit) or not math.isfinite(upper_limit) or upper_limit == 0:
            upper_limit = None

    denominator = upper_limit if upper_limit is not None else value_a + float_info.epsilon
    # Adding sys.float_info.epsilon to value_a to prevent division by zero if value_a is zero.

    delta_percentage = (100 * (value_a - value_b)) / denominator
    return delta_percentage


QUANTILES = {
    1: "all",
    2: "half",
    3: "tertile",
    4: "quartile",
    5: "quintile",
    6: "sextile",
    7: "septile",
    8: "octile",
    10: "decile",
    100: "percentile",
}


def get_quantile(n_buckets: int) -> str:
    return QUANTILES.get(n_buckets, "quantile")


def ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = ["th", "st", "nd", "rd", "th"][min(n % 10, 4)]
    return str(n) + suffix
