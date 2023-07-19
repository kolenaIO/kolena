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

from .test_evaluator_multiclass_test_sample_metrics import TEST_CONFIGURATIONS
from .test_evaluator_multiclass_test_sample_metrics import TEST_DATA
from kolena._experimental.object_detection import ObjectDetectionEvaluator
from kolena._experimental.object_detection import TestCase
from kolena.workflow import Plot
from kolena.workflow.plot import ConfusionMatrix
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
                                0.7924528301886793,
                                0.7735849056603774,
                                0.7547169811320755,
                                0.7169811320754716,
                                0.6981132075471698,
                                0.6509433962264151,
                                0.6320754716981132,
                                0.6132075471698113,
                                0.5943396226415094,
                                0.5377358490566038,
                                0.4716981132075472,
                                0.42452830188679247,
                                0.37735849056603776,
                                0.29245283018867924,
                                0.27358490566037735,
                                0.11320754716981132,
                                0.09433962264150944,
                                0.0660377358490566,
                                0.0,
                            ],
                            y=[
                                0.84,
                                0.845360824742268,
                                0.8421052631578947,
                                0.8351648351648352,
                                0.8314606741573034,
                                0.8214285714285714,
                                0.8170731707317073,
                                0.8125,
                                0.8076923076923077,
                                0.7916666666666666,
                                0.78125,
                                0.7627118644067796,
                                0.7407407407407407,
                                0.6888888888888889,
                                0.7073170731707317,
                                0.6,
                                0.7142857142857143,
                                0.7,
                                0.7,
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
                            x=[
                                0.0,
                                0.001,
                                0.01,
                                0.05,
                                0.1,
                                0.2,
                                0.25,
                                0.3,
                                0.4,
                                0.5,
                                0.6,
                                0.7,
                                0.8,
                                0.85,
                                0.88,
                                0.9,
                                0.95,
                                0.99,
                                1.0,
                            ],
                            y=[
                                0.8155339805825242,
                                0.8078817733990147,
                                0.7960199004975125,
                                0.7715736040609137,
                                0.758974358974359,
                                0.7263157894736841,
                                0.7127659574468085,
                                0.6989247311827956,
                                0.6847826086956521,
                                0.6404494382022472,
                                0.588235294117647,
                                0.5454545454545454,
                                0.5,
                                0.4105960264900662,
                                0.4161073825503355,
                                0.39455782312925164,
                                0.1904761904761905,
                                0.16666666666666669,
                                0.1206896551724138,
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
            "Threshold: Fixed(0.5) by class, IoU: 0.5, confidence ≥ 0.0",
            [
                CurvePlot(
                    title="F1-Score vs. Confidence Threshold Per Class",
                    x_label="Confidence Threshold",
                    y_label="F1-Score",
                    curves=[
                        Curve(
                            x=[0.001, 0.01, 0.4, 0.6, 0.7, 0.8, 0.85, 0.88, 0.9, 0.95, 0.99, 1.0],
                            y=[
                                0.7586206896551724,
                                0.7294117647058823,
                                0.6987951807228915,
                                0.6666666666666666,
                                0.6329113924050633,
                                0.6493506493506495,
                                0.5753424657534246,
                                0.5915492957746479,
                                0.5507246376811594,
                                0.3728813559322034,
                                0.3272727272727273,
                                0.2641509433962264,
                            ],
                            label="a",
                        ),
                        Curve(
                            x=[0.0, 0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.7, 0.9, 0.95, 0.99, 1.0],
                            y=[
                                0.3508771929824562,
                                0.3636363636363636,
                                0.3703703703703704,
                                0.3846153846153846,
                                0.4,
                                0.4166666666666667,
                                0.34782608695652173,
                                0.3,
                                0.3076923076923077,
                                0.3157894736842105,
                                0.0,
                                0.0,
                                0.0,
                            ],
                            label="b",
                        ),
                        Curve(
                            x=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9],
                            y=[
                                0.8695652173913044,
                                0.8181818181818181,
                                0.761904761904762,
                                0.7000000000000001,
                                0.631578947368421,
                                0.5555555555555556,
                                0.47058823529411764,
                                0.375,
                                0.14285714285714288,
                            ],
                            label="c",
                        ),
                        Curve(
                            x=[0.0, 0.1, 0.6, 0.7, 0.8],
                            y=[0.588235294117647, 0.5, 0.4, 0.28571428571428575, 0.15384615384615383],
                            label="e",
                        ),
                    ],
                    x_config=None,
                    y_config=None,
                ),
                CurvePlot(
                    title="Precision vs. Recall Per Class",
                    x_label="Recall",
                    y_label="Precision",
                    curves=[
                        Curve(
                            x=[
                                0.75,
                                0.7045454545454546,
                                0.6590909090909091,
                                0.6136363636363636,
                                0.5681818181818182,
                                0.4772727272727273,
                                0.4318181818181818,
                                0.25,
                                0.20454545454545456,
                                0.1590909090909091,
                                0.0,
                            ],
                            y=[
                                0.7674418604651163,
                                0.7560975609756098,
                                0.7435897435897436,
                                0.7297297297297297,
                                0.7142857142857143,
                                0.7241379310344828,
                                0.76,
                                0.7333333333333333,
                                0.8181818181818182,
                                0.7777777777777778,
                                0.7777777777777778,
                            ],
                            label="a",
                        ),
                        Curve(
                            x=[
                                0.4166666666666667,
                                0.3333333333333333,
                                0.25,
                                0.0,
                            ],
                            y=[
                                0.30303030303030304,
                                0.36363636363636365,
                                0.375,
                                0.0,
                            ],
                            label="b",
                        ),
                        Curve(
                            x=[
                                0.7692307692307693,
                                0.6923076923076923,
                                0.6153846153846154,
                                0.5384615384615384,
                                0.46153846153846156,
                                0.38461538461538464,
                                0.3076923076923077,
                                0.23076923076923078,
                                0.07692307692307693,
                                0.0,
                            ],
                            y=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                            label="c",
                        ),
                        Curve(x=[1.0, 0.0], y=[1.0, 1.0], label="d"),
                        Curve(
                            x=[0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
                            y=[
                                0.7142857142857143,
                                0.6666666666666666,
                                0.6,
                                0.5,
                                0.3333333333333333,
                                0.3333333333333333,
                            ],
                            label="e",
                        ),
                    ],
                    x_config=None,
                    y_config=None,
                ),
                ConfusionMatrix(
                    title="Confusion Matrix",
                    labels=["a", "b", "c", "d", "e"],
                    matrix=[[27, 2, 0, 0, 0], [0, 8, 0, 0, 0], [0, 0, 10, 0, 0], [0, 0, 0, 2, 0], [0, 1, 0, 0, 3]],
                    x_label="Predicted",
                    y_label="Actual",
                ),
            ],
        ),
        (
            "Threshold: F1-Optimal by class, IoU: 0.5, confidence ≥ 0.1",
            [
                CurvePlot(
                    title="F1-Score vs. Confidence Threshold Per Class",
                    x_label="Confidence Threshold",
                    y_label="F1-Score",
                    curves=[
                        Curve(
                            x=[0.4, 0.6, 0.7, 0.8, 0.85, 0.88, 0.9, 0.95, 0.99, 1.0],
                            y=[
                                0.6987951807228915,
                                0.6666666666666666,
                                0.6329113924050633,
                                0.6493506493506495,
                                0.5753424657534246,
                                0.5915492957746479,
                                0.5507246376811594,
                                0.3728813559322034,
                                0.3272727272727273,
                                0.2641509433962264,
                            ],
                            label="a",
                        ),
                        Curve(
                            x=[0.1, 0.25, 0.4, 0.5, 0.6, 0.7, 0.9, 0.95, 0.99, 1.0],
                            y=[
                                0.3846153846153846,
                                0.4,
                                0.4166666666666667,
                                0.34782608695652173,
                                0.3,
                                0.3076923076923077,
                                0.3157894736842105,
                                0.0,
                                0.0,
                                0.0,
                            ],
                            label="b",
                        ),
                        Curve(
                            x=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9],
                            y=[
                                0.8181818181818181,
                                0.761904761904762,
                                0.7000000000000001,
                                0.631578947368421,
                                0.5555555555555556,
                                0.47058823529411764,
                                0.375,
                                0.14285714285714288,
                            ],
                            label="c",
                        ),
                        Curve(
                            x=[0.1, 0.6, 0.7, 0.8],
                            y=[0.5, 0.4, 0.28571428571428575, 0.15384615384615383],
                            label="e",
                        ),
                    ],
                    x_config=None,
                    y_config=None,
                ),
                CurvePlot(
                    title="Precision vs. Recall Per Class",
                    x_label="Recall",
                    y_label="Precision",
                    curves=[
                        Curve(
                            x=[
                                0.6590909090909091,
                                0.6136363636363636,
                                0.5681818181818182,
                                0.4772727272727273,
                                0.4318181818181818,
                                0.25,
                                0.20454545454545456,
                                0.1590909090909091,
                                0.0,
                            ],
                            y=[
                                0.7435897435897436,
                                0.7297297297297297,
                                0.7142857142857143,
                                0.7241379310344828,
                                0.76,
                                0.7333333333333333,
                                0.8181818181818182,
                                0.7777777777777778,
                                0.7777777777777778,
                            ],
                            label="a",
                        ),
                        Curve(
                            x=[
                                0.4166666666666667,
                                0.3333333333333333,
                                0.25,
                                0.0,
                            ],
                            y=[
                                0.35714285714285715,
                                0.36363636363636365,
                                0.375,
                                0.0,
                            ],
                            label="b",
                        ),
                        Curve(
                            x=[
                                0.6923076923076923,
                                0.6153846153846154,
                                0.5384615384615384,
                                0.46153846153846156,
                                0.38461538461538464,
                                0.3076923076923077,
                                0.23076923076923078,
                                0.07692307692307693,
                                0.0,
                            ],
                            y=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                            label="c",
                        ),
                        Curve(x=[1.0, 0.0], y=[1.0, 1.0], label="d"),
                        Curve(
                            x=[0.4, 0.3, 0.2, 0.1, 0.0],
                            y=[0.6666666666666666, 0.6, 0.5, 0.3333333333333333, 0.3333333333333333],
                            label="e",
                        ),
                    ],
                    x_config=None,
                    y_config=None,
                ),
                ConfusionMatrix(
                    title="Confusion Matrix",
                    labels=["a", "b", "c", "d", "e"],
                    matrix=[[29, 2, 0, 0, 0], [0, 10, 0, 0, 0], [0, 0, 18, 0, 0], [0, 0, 0, 2, 0], [0, 1, 0, 0, 4]],
                    x_label="Predicted",
                    y_label="Actual",
                ),
            ],
        ),
    ],
)
def test__object_detection__multiclass__compute_test_case_plots__advanced(
    config_name: str,
    expected: Optional[List[Plot]],
) -> None:
    config = TEST_CONFIGURATIONS[config_name]
    eval = ObjectDetectionEvaluator(configurations=[config])

    test_case = TestCase(
        with_test_prefix(f"complete {config_name} {str(uuid.uuid4())}"),
        test_samples=[(ts, gt) for _, data in TEST_DATA.items() for ts, gt, _ in data],
        reset=True,
    )

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
