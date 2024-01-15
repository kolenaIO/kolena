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
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import evaluate
from question_answering.utils import compute_metric_bar_plot
from question_answering.utils import compute_metric_vs_metric_plot
from question_answering.utils import compute_score_distribution_plot
from question_answering.utils import mean_metric
from question_answering.workflow import GroundTruth
from question_answering.workflow import Inference
from question_answering.workflow import TestCase
from question_answering.workflow import TestCaseMetrics
from question_answering.workflow import TestSample
from question_answering.workflow import TestSampleMetrics
from question_answering.workflow import TestSuiteMetrics
from question_answering.workflow import ThresholdConfiguration
from tqdm import tqdm

from kolena.workflow import EvaluationResults
from kolena.workflow import Plot
from kolena.workflow import TestCases


bertscore = evaluate.load("bertscore")
rouge = evaluate.load("rouge")


def compute_metrics(norm_gt_answer: str, norm_inf_answer: str) -> Dict[str, float]:
    bertscore_results = bertscore.compute(
        predictions=[norm_inf_answer],
        references=[norm_gt_answer],
        lang="en",
        model_type="distilbert-base-uncased",
    )

    rouge_results = rouge.compute(
        predictions=[norm_inf_answer],
        references=[norm_gt_answer],
        rouge_types=["rouge1", "rouge2", "rougeL"],
    )

    return {
        "BERT_prec": bertscore_results["precision"][0],
        "BERT_rec": bertscore_results["recall"][0],
        "BERT_f1": bertscore_results["f1"][0],
        "ROUGE_1": rouge_results["rouge1"],
        "ROUGE_2": rouge_results["rouge2"],
        "ROUGE_L": rouge_results["rougeL"],
    }


def compute_test_sample_metrics(
    gt: GroundTruth,
    inf: Inference,
    configuration: ThresholdConfiguration,
) -> TestSampleMetrics:
    results = compute_metrics(gt.clean_answer, inf.clean_answer)
    custom_metric = round((results["BERT_f1"] + results["ROUGE_1"]) / 2, 3)
    return TestSampleMetrics(
        is_correct=True if custom_metric >= configuration.threshold else False,
        BERT_prec=results["BERT_prec"],
        BERT_rec=results["BERT_rec"],
        BERT_f1=results["BERT_f1"],
        MEAN_METRIC=custom_metric,
        ROUGE_1=results["ROUGE_1"],
        ROUGE_2=results["ROUGE_2"],
        ROUGE_L=results["ROUGE_L"],
    )


def compute_test_case_metrics(
    metrics: List[TestSampleMetrics],
) -> TestCaseMetrics:
    return TestCaseMetrics(
        n_correct=sum([1 for metric in metrics if metric.is_correct]),
        n_incorrect=sum([1 for metric in metrics if not metric.is_correct]),
        BERT_prec=mean_metric("BERT_prec", metrics),
        BERT_rec=mean_metric("BERT_rec", metrics),
        BERT_f1=mean_metric("BERT_f1", metrics),
        MEAN_METRIC=mean_metric("MEAN_METRIC", metrics),
        ROUGE_1=mean_metric("ROUGE_1", metrics),
        ROUGE_2=mean_metric("ROUGE_2", metrics),
        ROUGE_L=mean_metric("ROUGE_L", metrics),
    )


def compute_test_case_plots(
    test_samples: List[TestSample],
    inferences: List[Inference],
    metrics: List[TestSampleMetrics],
) -> List[Optional[Plot]]:
    metrics_of_interest = ["BERT_f1", "ROUGE_1", "MEAN_METRIC"]
    metric_values = [mean_metric(metric, metrics) for metric in metrics_of_interest]

    plots: List[Optional[Plot]] = [
        compute_metric_bar_plot(metrics_of_interest, metric_values),
        compute_score_distribution_plot("BERT_f1", metrics, (0, 1, 101)),
        compute_score_distribution_plot("ROUGE_1", metrics, (0, 1, 101)),
        compute_score_distribution_plot("ROUGE_L", metrics, (0, 1, 101)),
        compute_score_distribution_plot("MEAN_METRIC", metrics, (0, 1, 101)),
        compute_metric_vs_metric_plot("wc_answer", "BERT_f1", inferences, metrics, (0, 50, 26)),
        compute_metric_vs_metric_plot("wc_answer", "ROUGE_1", inferences, metrics, (0, 50, 26)),
        compute_metric_vs_metric_plot("turn", "ROUGE_1", test_samples, metrics, (0, 20, 26)),
        compute_metric_vs_metric_plot("turn", "BERT_f1", test_samples, metrics, (0, 20, 26)),
    ]

    return [plot for plot in plots if plot is not None]


def compute_test_suite_metrics(
    test_samples: List[TestSample],
    test_case_metrics: List[Tuple[TestCase, TestCaseMetrics]],
) -> TestSuiteMetrics:
    n_correct = sum([tcm.n_correct for _, tcm in test_case_metrics])
    n_questions = n_correct + sum([tcm.n_incorrect for _, tcm in test_case_metrics])
    unique_stories = {ts.data_id for ts in test_samples}

    return TestSuiteMetrics(
        n_stories=len(unique_stories),
        n_questions=n_questions,
        n_correct=n_correct,
        overall_accuracy=round(n_correct / n_questions, 3),
    )


def evaluate_question_answering(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
    configuration: ThresholdConfiguration,
) -> EvaluationResults:
    # compute sample-level metrics for each sample
    test_sample_metrics = [
        compute_test_sample_metrics(gt, inf, configuration)
        for gt, inf in tqdm(zip(ground_truths, inferences), total=len(ground_truths))
    ]

    # compute aggregate metrics across all test cases using `test_cases.iter(...)`
    all_test_case_metrics: List[Tuple[TestCase, TestCaseMetrics]] = []
    all_test_case_plots: List[Tuple[TestCase, List[Optional[Plot]]]] = []
    for test_case, ts, gt, inf, tsm in test_cases.iter(test_samples, ground_truths, inferences, test_sample_metrics):
        all_test_case_metrics.append((test_case, compute_test_case_metrics(tsm)))
        all_test_case_plots.append((test_case, compute_test_case_plots(ts, inf, tsm)))

    test_suite_metrics = compute_test_suite_metrics(test_samples, all_test_case_metrics)

    # if desired, compute and add `plots_test_case` and `metrics_test_suite` to this `EvaluationResults`
    return EvaluationResults(
        metrics_test_sample=list(zip(test_samples, test_sample_metrics)),
        metrics_test_case=all_test_case_metrics,  # type: ignore
        plots_test_case=all_test_case_plots,  # type: ignore
        metrics_test_suite=test_suite_metrics,
    )
