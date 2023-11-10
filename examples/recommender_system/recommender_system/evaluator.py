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
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from recommender_system.metrics import avg_precision_at_k
from recommender_system.metrics import mean_avg_precision_at_k
from recommender_system.metrics import mrr_at_k
from recommender_system.metrics import precision_at_k
from recommender_system.metrics import recall_at_k
from recommender_system.workflow import GroundTruth
from recommender_system.workflow import Inference
from recommender_system.workflow import RecommenderConfiguration
from recommender_system.workflow import TestCase
from recommender_system.workflow import TestCaseMetrics
from recommender_system.workflow import TestSample
from recommender_system.workflow import TestSampleMetrics

from kolena.workflow import EvaluationResults
from kolena.workflow import Plot
from kolena.workflow import TestCases


def compute_per_sample(
    ground_truth: GroundTruth,
    inference: Inference,
    configuration: RecommenderConfiguration,
) -> TestSampleMetrics:
    ratings = np.array([movie.score for movie in ground_truth.rated_movies])
    predictions = np.array([movie.score for movie in inference.recommendations])
    pk = precision_at_k(ratings, predictions, configuration.k)
    rk = recall_at_k(ratings, predictions, configuration.k)

    return TestSampleMetrics(
        RMSE=np.sqrt(((predictions - ratings) ** 2).mean()),
        MAE=np.abs(predictions - ratings).mean(),
        MAP=avg_precision_at_k(ratings, predictions, configuration.k),
        MRR=mrr_at_k(ratings, predictions, configuration.k),
        NDCG=mean_avg_precision_at_k(ratings, predictions, configuration.k),
        F1_k=2 * pk * rk / (pk + rk),
        precision_k=precision_at_k(ratings, predictions, configuration.k),
        recall_k=recall_at_k(ratings, predictions, configuration.k),
        avg_Δ_rating=np.mean([predictions - ratings]),
    )


def compute_test_case_metrics(
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    metrics: List[TestSampleMetrics],
) -> TestCaseMetrics:
    return TestCaseMetrics(
        AvgRMSE=np.mean([tsm.RMSE for tsm in metrics]),
        AvgMAE=np.mean([tsm.MAE for tsm in metrics]),
        AvgR2=np.mean([tsm.R2 for tsm in metrics]),
        AvgMAP=np.mean([tsm.MAP for tsm in metrics]),
        AvgMRR=np.mean([tsm.MRR for tsm in metrics]),
        AvgNDCG=np.mean([tsm.NDCG for tsm in metrics]),
        AvgPrecision_k=np.mean([tsm.precision_k for tsm in metrics]),
        AvgRecall_k=np.mean([tsm.recall_k for tsm in metrics]),
    )


def compute_test_case_plots(
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    metrics: List[TestSampleMetrics],
    configuration: RecommenderConfiguration,
) -> Optional[List[Plot]]:
    return NotImplementedError


def evaluate_recommender(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
    configuration: RecommenderConfiguration,
) -> EvaluationResults:
    # compute per-sample metrics for each test sample
    test_sample_metrics = [compute_per_sample(gt, inf, configuration) for gt, inf in zip(ground_truths, inferences)]

    # compute aggregate metrics across all test cases using `test_cases.iter(...)`
    all_test_case_metrics: List[Tuple[TestCase, TestCaseMetrics]] = []
    all_test_case_plots: List[Tuple[TestCase, List[Plot]]] = []
    for test_case, ts, gt, inf, tsm in test_cases.iter(test_samples, ground_truths, inferences, test_sample_metrics):
        all_test_case_metrics.append((test_case, compute_test_case_metrics(gt, inf, tsm)))
        all_test_case_plots.append((test_case, compute_test_case_plots(gt, inf, tsm, configuration)))

    return EvaluationResults(
        metrics_test_sample=list(zip(test_samples, test_sample_metrics)),
        metrics_test_case=all_test_case_metrics,
        plots_test_case=all_test_case_plots,
    )
