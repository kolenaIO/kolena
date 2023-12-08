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
import ast
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from question_answering.truthful_qa.workflow import GroundTruth
from question_answering.truthful_qa.workflow import Inference
from question_answering.truthful_qa.workflow import TestCase
from question_answering.truthful_qa.workflow import TestCaseMetrics
from question_answering.truthful_qa.workflow import TestSample
from question_answering.truthful_qa.workflow import TestSampleMetrics
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

from kolena.workflow import Curve
from kolena.workflow import CurvePlot
from kolena.workflow import EvaluationResults
from kolena.workflow import Plot
from kolena.workflow import TestCases

dfs_selfcheck = {}
dfs_uncertainty = {}
dfs_consistency = {}


def is_valid(inf: Inference) -> bool:
    return not (inf.missing_answer or inf.answer == "nan")


def compute_uncertainty(ts: TestSample, inf: Inference) -> Dict[str, float]:
    def get_uncertainty_metrics(locator: str) -> pd.DataFrame:
        if locator not in dfs_uncertainty:
            df = pd.read_csv(locator)
            df = df.dropna()
            dfs_uncertainty[locator] = df

        return dfs_uncertainty[locator]

    df_uncertainty = get_uncertainty_metrics(inf.probabilities_metrics.locator)
    df_question = df_uncertainty[df_uncertainty["question"] == ts.question]
    if len(df_question) > 0:
        uncertainty_scores = df_question.iloc[0]
    else:
        return None

    return dict(
        average_logprob=uncertainty_scores.average_logprob,
        average_entropy=uncertainty_scores.average_entropy,
        min_logprob=uncertainty_scores.min_logprob,
        max_entropy=uncertainty_scores.max_entropy,
    )


def compute_selfcheck_scores(ts: TestSample, inf: Inference) -> Dict[str, float]:
    def get_selfcheck_metrics(locator: str) -> pd.DataFrame:
        if locator not in dfs_selfcheck:
            df = pd.read_csv(locator)
            df = df.dropna()
            dfs_selfcheck[locator] = df

        return dfs_selfcheck[locator]

    df_selfcheck = get_selfcheck_metrics(inf.selfcheck_metrics.locator)
    df_question = df_selfcheck[df_selfcheck["question"] == ts.question]
    if len(df_question) > 0:
        selfcheck_scores = df_question.iloc[0]
    else:
        return None

    return dict(
        selfcheck_bertscore=selfcheck_scores.selfcheck_bertscore,
        selfcheck_ngram=selfcheck_scores.selfcheck_ngram,
    )


def compute_selfcheck_prompt_scores(ts: TestSample, inf: Inference) -> Dict[str, float]:
    def get_selfcheck_prompt_metrics(locator: str) -> pd.DataFrame:
        if locator not in dfs_consistency:
            df = pd.read_csv(locator)
            df = df.dropna()
            dfs_consistency[locator] = df

        return dfs_consistency[locator]

    df_consistency = get_selfcheck_prompt_metrics(inf.consistency_metrics.locator)
    df_question = df_consistency[df_consistency["question"] == ts.question]
    if len(df_question) > 0:
        consistency_results = df_question.iloc[0]["consistency"]
        consistency_results = ast.literal_eval(consistency_results)
        consistency_score = np.mean([consistency for consistency, _ in consistency_results])
        reasons = [reason for _, reason in consistency_results if reason != ""]
    else:
        return None

    return dict(
        consistency_score=consistency_score,
        reasons=reasons,
    )


def compute_test_sample_metrics(
    ts: TestSample,
    gt: GroundTruth,
    inf: Inference,
) -> TestSampleMetrics:
    if not is_valid(inf):
        return TestSampleMetrics(
            fail_to_answer=True,
        )

    uncertainty_scores = compute_uncertainty(ts, inf)
    selfcheck_scores = compute_selfcheck_scores(ts, inf)
    consistency_scores = compute_selfcheck_prompt_scores(ts, inf)

    return TestSampleMetrics(
        fail_to_answer=False,
        is_hallucination_by_logprob=uncertainty_scores["min_logprob"] > 5.0 if uncertainty_scores is not None else None,
        is_hallucination_by_entropy=uncertainty_scores["average_entropy"] > 0.5
        if uncertainty_scores is not None
        else None,
        average_logprob=uncertainty_scores["average_logprob"] if uncertainty_scores is not None else None,
        average_entropy=uncertainty_scores["average_entropy"] if uncertainty_scores is not None else None,
        min_logprob=uncertainty_scores["min_logprob"] if uncertainty_scores is not None else None,
        max_entropy=uncertainty_scores["max_entropy"] if uncertainty_scores is not None else None,
        is_hallucination_by_selfcheck_bertscore=selfcheck_scores["selfcheck_bertscore"] > 0.5
        if selfcheck_scores is not None
        else None,
        is_hallucination_by_selfcheck_ngram=selfcheck_scores["selfcheck_ngram"] > 0.5
        if selfcheck_scores is not None
        else None,
        selfcheck_bertscore=selfcheck_scores["selfcheck_bertscore"] if selfcheck_scores is not None else None,
        selfcheck_ngram=selfcheck_scores["selfcheck_ngram"] if selfcheck_scores is not None else None,
        is_hallucination_by_selfcheck_prompt=consistency_scores["consistency_score"] <= 0.75
        if consistency_scores is not None
        else None,
        selfcheck_prompt=consistency_scores["consistency_score"] if consistency_scores is not None else None,
        selfcheck_prompt_reasons=consistency_scores["reasons"] if consistency_scores is not None else None,
    )


def compute_test_case_metrics(
    metrics: List[TestSampleMetrics],
    inferences: List[Inference],
) -> TestCaseMetrics:
    return TestCaseMetrics(
        Questions=len(metrics),
        Failures=np.sum([m.fail_to_answer for m in metrics]),
        FactualityScoreLogProb=np.mean([not m.fail_to_answer and not m.is_hallucination_by_logprob for m in metrics]),
        FactualityScoreEntropy=np.mean([not m.fail_to_answer and not m.is_hallucination_by_entropy for m in metrics]),
        FactualityScoreSelfcheckBert=np.mean(
            [not m.fail_to_answer and not m.is_hallucination_by_selfcheck_bertscore for m in metrics],
        ),
        FactualityScoreSelfcheckNGram=np.mean(
            [not m.fail_to_answer and not m.is_hallucination_by_selfcheck_ngram for m in metrics],
        ),
        MetricsAccuracyLogProb=np.mean(
            [
                inf.is_hallucination == m.is_hallucination_by_logprob
                for m, inf in zip(metrics, inferences)
                if inf.is_hallucination is not None
            ],
        ),
        MetricsAccuracyEntropy=np.mean(
            [
                inf.is_hallucination == m.is_hallucination_by_entropy
                for m, inf in zip(metrics, inferences)
                if inf.is_hallucination is not None
            ],
        ),
        MetricsAccuracySelfcheckBert=np.mean(
            [
                inf.is_hallucination == m.is_hallucination_by_selfcheck_bertscore
                for m, inf in zip(metrics, inferences)
                if inf.is_hallucination is not None
            ],
        ),
        MetricsAccuracySelfcheckNGram=np.mean(
            [
                inf.is_hallucination == m.is_hallucination_by_selfcheck_ngram
                for m, inf in zip(metrics, inferences)
                if inf.is_hallucination is not None
            ],
        ),
        MetricsAccuracySelfcheckPrompt=np.mean(
            [
                inf.is_hallucination == m.is_hallucination_by_selfcheck_prompt
                for m, inf in zip(metrics, inferences)
                if inf.is_hallucination is not None
            ],
        ),
    )


def compute_test_case_plots(
    metrics: List[TestSampleMetrics],
    inferences: List[Inference],
) -> Optional[List[Plot]]:
    metric_types = ["selfcheck_prompt"]

    curves = []
    for metric_type in metric_types:
        y_true = []
        y_pred = []
        for m, inf in zip(metrics, inferences):
            metric_value = getattr(m, metric_type)
            if inf.missing_answer or inf.is_hallucination is None or metric_value is None:
                continue
            y_true.append(not inf.is_hallucination)
            y_pred.append(metric_value)

        p, r, t = precision_recall_curve(y_true, y_pred, pos_label=True)

        curves.append(
            Curve(
                x=r.tolist(),
                y=p.tolist(),
                label=metric_type,
                extra={"T": t.tolist() + [t[-1]]},
            ),
        )
    return [
        CurvePlot(
            title="Precision-Recall Curve",
            x_label="Recall",
            y_label="Precision",
            curves=curves,
        ),
    ]


def evaluate_question_answering(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
) -> EvaluationResults:
    # compute sample-level metrics for each sample
    test_sample_metrics = [
        compute_test_sample_metrics(ts, gt, inf)
        for ts, gt, inf in tqdm(zip(test_samples, ground_truths, inferences), total=len(ground_truths))
    ]

    # compute aggregate metrics across all test cases using `test_cases.iter(...)`
    all_test_case_metrics: List[Tuple[TestCase, TestCaseMetrics]] = []
    all_test_case_plots: List[Tuple[TestCase, List[Plot]]] = []
    for test_case, ts, gt, inf, tsm in test_cases.iter(test_samples, ground_truths, inferences, test_sample_metrics):
        all_test_case_metrics.append((test_case, compute_test_case_metrics(tsm, inf)))
        all_test_case_plots.append((test_case, compute_test_case_plots(tsm, inf)))

    # if desired, compute and add `plots_test_case` and `metrics_test_suite` to this `EvaluationResults`
    return EvaluationResults(
        metrics_test_sample=list(zip(test_samples, test_sample_metrics)),
        metrics_test_case=all_test_case_metrics,
        plots_test_case=all_test_case_plots,
    )
