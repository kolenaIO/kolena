from collections import defaultdict
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from .workflow import GroundTruth
from .workflow import Inference
from .workflow import TestCase
from .workflow import TestSample
from kolena.workflow import Curve
from kolena.workflow import CurvePlot
from kolena.workflow import EvaluationResults
from kolena.workflow import Evaluator
from kolena.workflow import EvaluatorConfiguration
from kolena.workflow import MetricsTestCase as BaseMetricsTestCase
from kolena.workflow import MetricsTestSample as BaseMetricsTestSample
from kolena.workflow import Plot


@dataclass(frozen=True)
class MetricsTestSample(BaseMetricsTestSample):
    error: Optional[float]  # absolute error
    fail_to_detect: bool = False


@dataclass(frozen=True)
class MetricsTestCase(BaseMetricsTestCase):
    n_infer_fail: int
    mae: Optional[float] = None  # mean absolute error
    rmse: Optional[float] = None  # root mean squared error
    failure_rate_err_gt_5: Optional[float] = None


def evaluate_age_estimation() -> EvaluationResults:
    ...


class AgeEstimationEvaluator(Evaluator):
    def compute_test_sample_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        configuration: Optional[EvaluatorConfiguration] = None,
    ) -> List[Tuple[TestSample, MetricsTestSample]]:
        return [
            (
                ts,
                MetricsTestSample(
                    error=abs(inf.age - gt.age) if inf.age is not None else None,
                    fail_to_detect=inf.age is None,
                ),
            )
            for ts, gt, inf in inferences
        ]

    def compute_test_case_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        metrics: List[MetricsTestSample],
        configuration: Optional[EvaluatorConfiguration] = None,
    ) -> MetricsTestCase:
        num_valid_predictions = sum(not metric.fail_to_detect for metric in metrics)
        if num_valid_predictions == 0:
            return MetricsTestCase(n_infer_fail=len(metrics))

        abs_errors = np.array([metric.error for metric in metrics if metric.error is not None])
        abs_errors_squared = np.array([ae**2 for ae in abs_errors])

        return MetricsTestCase(
            mae=abs_errors.mean(),
            rmse=np.sqrt(abs_errors_squared.mean()),
            n_infer_fail=len(metrics) - num_valid_predictions,
            failure_rate_err_gt_5=np.sum(abs_errors > 5) / float(num_valid_predictions),
        )

    def compute_test_case_plots(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        metrics: List[MetricsTestSample],
        configuration: Optional[EvaluatorConfiguration] = None,
    ) -> Optional[List[Plot]]:
        plots = []
        data = [mts.error for mts in metrics if mts.error is not None]
        y, x = np.histogram(data, bins=100, range=(0, 10))
        y = y / len(data)
        plots.append(
            CurvePlot(
                title="Distribution of Absolute Error",
                x_label="Absolute Error",
                y_label="Density",
                curves=[Curve(label="AE", x=x[1:].tolist(), y=y.tolist())],
            ),
        )

        mae_data = defaultdict(list)
        for mts, (_, gt, _) in zip(metrics, inferences):
            if mts.error is not None:
                mae_data[gt.age].append(mts.error)

        sorted_data = dict(sorted(mae_data.items()))
        x = list(sorted_data.keys())
        y = [sum(sorted_data[age]) / float(len(sorted_data[age])) for age in x]
        plots.append(
            CurvePlot(
                title="Distribution of Mean Absolute Error Across Target Age",
                x_label="Target Age",
                y_label="Mean Absolute Error",
                curves=[Curve(label="MAE", x=x, y=y)],
            ),
        )
        return plots
