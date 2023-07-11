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
import uuid
from typing import List
from typing import Optional

import pytest

from .test_evaluator_single_class_test_sample_metrics import TEST_CONFIGURATIONS
from .test_evaluator_single_class_test_sample_metrics import TEST_DATA
from kolena._experimental.object_detection import ObjectDetectionEvaluator
from kolena._experimental.object_detection import TestCase
from kolena.workflow import Plot
from kolena.workflow.plot import Curve
from kolena.workflow.plot import CurvePlot
from tests.integration.helper import with_test_prefix


@pytest.mark.metrics
@pytest.mark.parametrize(
    "config_name, expected",
    [
        (
            "Threshold: Fixed(0.3), IoU: 0.3, confidence ≥ 0.0",
            [
                CurvePlot(
                    title="Precision vs. Recall",
                    x_label="Recall",
                    y_label="Precision",
                    curves=[
                        Curve(
                            x=[
                                0.7142857142857143,
                                0.6857142857142857,
                                0.6571428571428571,
                                0.6285714285714286,
                                0.6,
                                0.5428571428571428,
                                0.5142857142857142,
                                0.22857142857142856,
                                0.02857142857142857,
                            ],
                            y=[
                                0.6944444444444444,
                                0.7058823529411765,
                                0.696969696969697,
                                0.6875,
                                0.6774193548387096,
                                0.6785714285714286,
                                0.6666666666666666,
                                0.5333333333333333,
                                0.3333333333333333,
                            ],
                            label=None,
                        ),
                    ],
                    x_config=None,
                    y_config=None,
                ),
                CurvePlot(
                    title="F1-Score vs. Confidence Threshold",
                    x_label="Confidence Threshold",
                    y_label="F1-Score",
                    curves=[
                        Curve(
                            x=[0.0, 0.001, 0.01, 0.4, 0.5, 0.6, 0.9, 0.99, 1.0],
                            y=[
                                0.7042253521126761,
                                0.6956521739130436,
                                0.676470588235294,
                                0.6567164179104478,
                                0.6363636363636364,
                                0.603174603174603,
                                0.5806451612903226,
                                0.32,
                                0.05263157894736842,
                            ],
                            label=None,
                        ),
                    ],
                    x_config=None,
                    y_config=None,
                ),
            ],
        ),
        (
            "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.0",
            [
                CurvePlot(
                    title="Precision vs. Recall",
                    x_label="Recall",
                    y_label="Precision",
                    curves=[
                        Curve(
                            x=[
                                0.6857142857142857,
                                0.6571428571428571,
                                0.6285714285714286,
                                0.6,
                                0.5714285714285714,
                                0.5142857142857142,
                                0.4857142857142857,
                                0.22857142857142856,
                                0.02857142857142857,
                            ],
                            y=[
                                0.6666666666666666,
                                0.6764705882352942,
                                0.6666666666666666,
                                0.65625,
                                0.6451612903225806,
                                0.6428571428571429,
                                0.6296296296296297,
                                0.5333333333333333,
                                0.3333333333333333,
                            ],
                            label=None,
                        ),
                    ],
                    x_config=None,
                    y_config=None,
                ),
                CurvePlot(
                    title="F1-Score vs. Confidence Threshold",
                    x_label="Confidence Threshold",
                    y_label="F1-Score",
                    curves=[
                        Curve(
                            x=[0.0, 0.001, 0.01, 0.4, 0.5, 0.6, 0.9, 0.99, 1.0],
                            y=[
                                0.676056338028169,
                                0.6666666666666666,
                                0.6470588235294118,
                                0.626865671641791,
                                0.606060606060606,
                                0.5714285714285714,
                                0.5483870967741936,
                                0.32,
                                0.05263157894736842,
                            ],
                            label=None,
                        ),
                    ],
                    x_config=None,
                    y_config=None,
                ),
            ],
        ),
        (
            "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.3",
            [
                CurvePlot(
                    title="Precision vs. Recall",
                    x_label="Recall",
                    y_label="Precision",
                    curves=[
                        Curve(
                            x=[
                                0.6,
                                0.5714285714285714,
                                0.5142857142857142,
                                0.4857142857142857,
                                0.22857142857142856,
                                0.02857142857142857,
                            ],
                            y=[
                                0.65625,
                                0.6451612903225806,
                                0.6428571428571429,
                                0.6296296296296297,
                                0.5333333333333333,
                                0.3333333333333333,
                            ],
                            label=None,
                        ),
                    ],
                    x_config=None,
                    y_config=None,
                ),
                CurvePlot(
                    title="F1-Score vs. Confidence Threshold",
                    x_label="Confidence Threshold",
                    y_label="F1-Score",
                    curves=[
                        Curve(
                            x=[0.4, 0.5, 0.6, 0.9, 0.99, 1.0],
                            y=[
                                0.626865671641791,
                                0.606060606060606,
                                0.5714285714285714,
                                0.5483870967741936,
                                0.32,
                                0.05263157894736842,
                            ],
                            label=None,
                        ),
                    ],
                    x_config=None,
                    y_config=None,
                ),
            ],
        ),
        (
            "Threshold: F1-Optimal, IoU: 0.5, confidence ≥ 0.1",
            [
                CurvePlot(
                    title="Precision vs. Recall",
                    x_label="Recall",
                    y_label="Precision",
                    curves=[
                        Curve(
                            x=[
                                0.6,
                                0.5714285714285714,
                                0.5142857142857142,
                                0.4857142857142857,
                                0.22857142857142856,
                                0.02857142857142857,
                            ],
                            y=[
                                0.65625,
                                0.6451612903225806,
                                0.6428571428571429,
                                0.6296296296296297,
                                0.5333333333333333,
                                0.3333333333333333,
                            ],
                            label=None,
                        ),
                    ],
                    x_config=None,
                    y_config=None,
                ),
                CurvePlot(
                    title="F1-Score vs. Confidence Threshold",
                    x_label="Confidence Threshold",
                    y_label="F1-Score",
                    curves=[
                        Curve(
                            x=[0.4, 0.5, 0.6, 0.9, 0.99, 1.0],
                            y=[
                                0.626865671641791,
                                0.606060606060606,
                                0.5714285714285714,
                                0.5483870967741936,
                                0.32,
                                0.05263157894736842,
                            ],
                            label=None,
                        ),
                    ],
                    x_config=None,
                    y_config=None,
                ),
            ],
        ),
    ],
)
def test__prebuilt__object__detection__single__class__compute__test__case__plots__advanced(
    config_name: str,
    expected: Optional[List[Plot]],
) -> None:
    config = TEST_CONFIGURATIONS[config_name]
    test_case = TestCase(
        with_test_prefix(f"complete {config_name} {str(uuid.uuid4())}"),
        test_samples=[(ts, gt) for _, data in TEST_DATA.items() for ts, gt, _ in data],
    )
    eval = ObjectDetectionEvaluator(configurations=[config])
    eval.compute_test_sample_metrics(
        test_case=test_case,
        inferences=[item for data in TEST_DATA.values() for item in data],
        configuration=config,
    )

    result = eval.compute_test_case_plots(
        test_case=test_case,
        inferences=[],
        metrics=[],
        configuration=config,
    )
    assert expected == result
