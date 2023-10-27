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
from rating_based_recommendation.utils import create_histogram
from rating_based_recommendation.workflow import GroundTruth
from rating_based_recommendation.workflow import Inference
from rating_based_recommendation.workflow import TestCase
from rating_based_recommendation.workflow import TestCaseMetrics
from rating_based_recommendation.workflow import TestSample
from rating_based_recommendation.workflow import TestSampleMetrics
from rating_based_recommendation.workflow import ThresholdConfiguration
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve

from kolena.workflow import ConfusionMatrix
from kolena.workflow import Curve
from kolena.workflow import CurvePlot
from kolena.workflow import EvaluationResults
from kolena.workflow import Plot
from kolena.workflow import TestCases
from kolena.workflow.metrics import accuracy
from kolena.workflow.metrics import f1_score
from kolena.workflow.metrics import precision
from kolena.workflow.metrics import recall


def compute_per_sample(
    ground_truth: GroundTruth,
    inference: Inference,
    configuration: ThresholdConfiguration,
) -> TestSampleMetrics:
    is_relevant = ground_truth.rating >= configuration.rating_threshold
    is_recommended = inference.pred_rating >= configuration.rating_threshold

    return TestSampleMetrics(
        is_correct=is_relevant == is_recommended,
        is_TP=is_relevant and is_recommended,
        is_FP=not is_relevant and is_recommended,
        is_FN=is_relevant and not is_recommended,
        is_TN=not is_relevant and not is_recommended,
        Δ_rating=inference.pred_rating - ground_truth.rating,
    )


def compute_test_case_metrics(
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    metrics: List[TestSampleMetrics],
) -> TestCaseMetrics:
    ratings = np.array([gt.rating for gt in ground_truths])
    preds = np.array([inf.pred_rating for inf in inferences])

    rmse = np.sqrt(((preds - ratings) ** 2).mean())
    mae = np.abs(preds - ratings).mean()

    tp = np.sum([tsm.is_TP for tsm in metrics])
    fp = np.sum([tsm.is_FP for tsm in metrics])
    fn = np.sum([tsm.is_FN for tsm in metrics])
    tn = np.sum([tsm.is_TN for tsm in metrics])

    high_rating = np.sum([r >= 5 for r in ratings])
    high_rating_fnr = (
        np.sum(
            [int(fn and gt.rating >= 5) for gt, fn in zip(ground_truths, [tsm.is_FN for tsm in metrics])],
        )
        / high_rating
        if high_rating != 0
        else 0.0
    )

    low_rating = np.sum([r <= 1 for r in ratings])
    low_rating_fpr = (
        np.sum(
            [int(fp and gt.rating <= 1) for gt, fp in zip(ground_truths, [tsm.is_FP for tsm in metrics])],
        )
        / low_rating
        if low_rating != 0
        else 0.0
    )

    return TestCaseMetrics(
        RMSE=rmse,
        MAE=mae,
        TP=tp,
        FP=fp,
        FN=fp,
        TN=tn,
        Accuracy=accuracy(tp, fp, fn, tn),
        Precision=precision(tp, fp),
        Recall=recall(tp, fn),
        F1=f1_score(tp, fp, fn),
        HighRatingFNR=high_rating_fnr,
        LowRatingFPR=low_rating_fpr,
    )


def compute_test_case_plots(
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    metrics: List[TestSampleMetrics],
    configuration: ThresholdConfiguration,
) -> Optional[List[Plot]]:
    plots = []

    tp = np.sum([tsm.is_TP for tsm in metrics])
    tn = np.sum([tsm.is_TN for tsm in metrics])
    fp = np.sum([tsm.is_FP for tsm in metrics])
    fn = np.sum([tsm.is_FN for tsm in metrics])

    plots.append(
        ConfusionMatrix(
            title="Rating Confusion Matrix",
            labels=["Recommended", "Not Recommended"],
            matrix=[[tp, fp], [fn, tn]],
        ),
    )

    plots.append(create_histogram(metrics))

    gts_binary_labels = [int(gt.rating >= configuration.rating_threshold) for gt in ground_truths]
    infs = [inf.pred_rating for inf in inferences]

    precision, recall, _ = precision_recall_curve(gts_binary_labels, infs)

    plots.append(
        CurvePlot(
            title="Precision vs. Recall",
            x_label="Recall",
            y_label="Precision",
            curves=[Curve(x=list(recall), y=list(precision))],
        ),
    )

    fpr, tpr, _ = roc_curve(gts_binary_labels, infs)
    roc_auc = auc(fpr, tpr)

    plots.append(
        CurvePlot(
            title="Receiver Operating Characteristic",
            x_label="False Positive Rate (FPR)",
            y_label="True Positive Rate (TPR)",
            curves=[Curve(x=list(fpr), y=list(tpr), label=f"AUC={roc_auc:.4f}")],
        ),
    )

    return plots


def evaluate_recommender(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
    configuration: ThresholdConfiguration,
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
