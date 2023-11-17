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
from collections import defaultdict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from recommender_system.metrics import avg_precision_at_k
from recommender_system.metrics import compute_classification_metrics
from recommender_system.metrics import compute_errors
from recommender_system.metrics import mean_avg_precision_at_k
from recommender_system.metrics import mrr
from recommender_system.workflow import GroundTruth
from recommender_system.workflow import Inference
from recommender_system.workflow import RecommenderConfiguration
from recommender_system.workflow import TestCase
from recommender_system.workflow import TestCaseMetrics
from recommender_system.workflow import TestSample
from recommender_system.workflow import TestSampleMetrics
from sklearn.metrics import ndcg_score

from kolena.workflow import BarPlot
from kolena.workflow import ConfusionMatrix
from kolena.workflow import EvaluationResults
from kolena.workflow import Plot
from kolena.workflow import TestCases


def compute_per_sample(
    ground_truth: GroundTruth,
    inference: Inference,
    configuration: RecommenderConfiguration,
) -> TestSampleMetrics:
    true_scores = np.asarray([[movie.score for movie in ground_truth.rated_movies]])
    pred_scores = np.asarray([[movie.score for movie in inference.recommendations]])
    ratings = [movie.id for movie in ground_truth.rated_movies]
    predictions = [movie.id for movie in inference.recommendations]

    k = configuration.k
    metrics = compute_classification_metrics(ground_truth, inference, configuration.threshold, k)
    rmse, mae = compute_errors(ground_truth, inference, k)

    return TestSampleMetrics(
        RMSE=rmse,
        MAE=mae,
        AP=avg_precision_at_k(ratings, predictions, k),
        MAP=mean_avg_precision_at_k(ratings, predictions, k),
        MRR=mrr(ratings, predictions, k),
        NDCG=ndcg_score(true_scores, pred_scores, k=k),
        F1_k=2 * metrics.pk * metrics.rk / (metrics.pk + metrics.rk) if (metrics.pk + metrics.rk) > 0 else 0,
        precision_k=metrics.pk,
        recall_k=metrics.rk,
        count_TP=metrics.tp,
        count_FP=metrics.fp,
        count_FN=metrics.fn,
        count_TN=metrics.tn,
    )


def compute_test_case_metrics(
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    metrics: List[TestSampleMetrics],
) -> TestCaseMetrics:
    return TestCaseMetrics(
        AvgRMSE=np.mean([tsm.RMSE for tsm in metrics]),
        AvgMAE=np.mean([tsm.MAE for tsm in metrics]),
        AvgAP=np.mean([tsm.AP for tsm in metrics]),
        AvgMAP=np.mean([tsm.MAP for tsm in metrics]),
        AvgMRR=np.mean([tsm.MRR for tsm in metrics]),
        AvgNDCG=np.mean([tsm.NDCG for tsm in metrics]),
        AvgPrecision_k=np.mean([tsm.precision_k for tsm in metrics]),
        AvgRecall_k=np.mean([tsm.recall_k for tsm in metrics]),
        TP=np.sum([tsm.count_TP for tsm in metrics]),
        FP=np.sum([tsm.count_FP for tsm in metrics]),
        FN=np.sum([tsm.count_FN for tsm in metrics]),
        TN=np.sum([tsm.count_TN for tsm in metrics]),
    )


def compute_test_case_plots(
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    metrics: List[TestSampleMetrics],
    configuration: RecommenderConfiguration,
) -> Optional[List[Plot]]:
    plots = []

    tp = np.sum([tsm.count_TP for tsm in metrics])
    fp = np.sum([tsm.count_FP for tsm in metrics])
    fn = np.sum([tsm.count_FN for tsm in metrics])
    tn = np.sum([tsm.count_TN for tsm in metrics])

    plots.append(
        ConfusionMatrix(
            title="Recommendation Confusion Matrix",
            labels=["Recommended", "Not Recommended"],
            matrix=[[tp, fp], [fn, tn]],
        ),
    )

    # novelty
    popularity_metrics = defaultdict(int)
    for gt in ground_truths:
        for movie in gt.rated_movies:
            popularity_metrics[movie.id] += 1
    sorted_popularity = dict(sorted(popularity_metrics.items(), key=lambda item: item[1], reverse=True))

    plots.append(
        BarPlot(
            title="Novelty",
            x_label="Movie",
            y_label="Popularity",
            labels=list(sorted_popularity.keys()),
            values=list(sorted_popularity.values()),
        ),
    )

    return plots


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
