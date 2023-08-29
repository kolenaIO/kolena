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
import re
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from workflow import Inference
from workflow import TestSampleMetrics

from kolena.workflow import AxisConfig
from kolena.workflow import BarPlot
from kolena.workflow import Curve
from kolena.workflow import CurvePlot
from kolena.workflow import Histogram


def get_readable(text: str) -> str:
    # no spaces before periods, only after
    return re.sub(r"\s+(\.)", r"\1", text)


def normalize_string(text: str) -> str:
    text = str(text)
    lower_text = text.lower()

    # remove punctuation except apostrophes and hyphens within words
    clean_text = re.sub(r"[^\w\s'-]", "", lower_text)

    # remove "the", "a", and "an"
    clean_text = re.sub(r"\b(the|a|an)\b", "", clean_text)

    # remove extra spaces
    normalized_text = re.sub(r"\s+", " ", clean_text).strip()

    return normalized_text


def mean_metric(metric: str, metrics: List[TestSampleMetrics]) -> float:
    return np.mean([getattr(m, metric) for m in metrics])


def compute_score_distribution_plot(
    score: str,
    metrics: List[Union[TestSampleMetrics, Inference]],
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


def compute_metric_vs_metric_plot(
    x_metric: str,
    y_metric: str,
    x_metrics: List[Union[TestSampleMetrics, Inference]],
    y_metrics: List[Union[TestSampleMetrics, Inference]],
    binning_info: Optional[Tuple[float, float, float]] = None,  # start, end, num
    x_logarithmic: bool = False,
    y_logarithmic: bool = False,
) -> Optional[CurvePlot]:
    y_values = [getattr(m, y_metric) for m in y_metrics]
    x_values = [getattr(m, x_metric) for m in x_metrics]

    if len(set(x_values)) < 2:
        return None

    if x_logarithmic:
        bins = list(np.logspace(*binning_info, base=2))
    else:
        bins = list(np.linspace(*binning_info))

    bins_centers: List[float] = []
    bins_values: List[float] = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i : i + 2]
        bin_values = [y for y, x in zip(y_values, x_values) if lo <= x < hi]
        if len(bin_values) > 0:
            bins_centers.append(lo + ((hi - lo) / 2))
            bins_values.append(np.mean(bin_values))

    return CurvePlot(
        title=f"{y_metric} vs. {x_metric}",
        x_label=f"{x_metric}",
        y_label=f"{y_metric}",
        curves=[Curve(x=bins_centers, y=bins_values)],
        x_config=AxisConfig(type="log") if x_logarithmic else None,
        y_config=AxisConfig(type="log") if y_logarithmic else None,
    )


def compute_metric_bar_plot(
    labels: List[str],
    values: List[float],
) -> BarPlot:
    return BarPlot(
        title="Performance Metrics Across Test Cases",
        x_label="Metrics",
        y_label="Performance",
        labels=labels,
        values=values,
    )
