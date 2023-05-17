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
from typing import Union

import evaluate
import numpy as np

from .workflow import GroundTruth
from .workflow import Inference
from .workflow import TestCase
from .workflow import TestCaseMetric
from .workflow import TestSample
from .workflow import TestSampleMetric
from .workflow import TestSuiteMetric
from kolena.workflow import Histogram
from kolena.workflow import Plot
from kolena.workflow.evaluator import AxisConfig
from kolena.workflow.evaluator import Curve
from kolena.workflow.evaluator import CurvePlot
from kolena.workflow.evaluator_function import EvaluationResults
from kolena.workflow.evaluator_function import TestCases

bertscore = evaluate.load("bertscore")
bleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")


def compute_test_sample_metrics(gt: GroundTruth, inf: Inference) -> TestSampleMetric:
    if not inf.is_failure:
        bertscore_results = bertscore.compute(
            predictions=[inf.summary],
            references=[gt.summary],
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli",
        )
        bleu_results = bleu.compute(predictions=[inf.summary], references=[gt.summary])
        rouge_results = rouge.compute(
            predictions=[inf.summary],
            references=[gt.summary],
            rouge_types=["rouge1", "rouge2", "rougeL"],
        )

    return TestSampleMetric(
        BERT_prec=bertscore_results["precision"][0] if not inf.is_failure else 0.0,
        BERT_rec=bertscore_results["recall"][0] if not inf.is_failure else 0.0,
        BERT_f1=bertscore_results["f1"][0] if not inf.is_failure else 0.0,
        ROUGE_1=rouge_results["rouge1"] if not inf.is_failure else 0.0,
        ROUGE_2=rouge_results["rouge2"] if not inf.is_failure else 0.0,
        ROUGE_L=rouge_results["rougeL"] if not inf.is_failure else 0.0,
        BLEU=bleu_results["score"] / 100.0 if not inf.is_failure else 0.0,
        inf_to_gt_word_count=float(inf.word_count) / gt.word_count,
    )


def compute_aggregate_metrics(
    test_samples_metrics: List[TestSampleMetric],
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
) -> TestCaseMetric:
    def mean_metric_ignoring_failures(metric: str) -> float:
        return np.mean([getattr(m, metric) for m, inf in zip(test_samples_metrics, inferences) if not inf.is_failure])

    num_failures = sum([inf.is_failure for inf in inferences])
    return TestCaseMetric(
        n_failures=num_failures,
        failure_rate=num_failures / len(test_samples_metrics),
        total_cost=np.sum([inf.cost for inf in inferences]),
        avg_cost=np.mean([inf.cost for inf in inferences]),
        avg_inference_time=np.mean([inf.inference_time for inf in inferences if inf.inference_time is not None]),
        avg_wc_input=np.mean([ts.word_count for ts in test_samples]),
        avg_wc_gt=np.mean([gt.word_count for gt in ground_truths]),
        avg_wc_inf=np.mean([inf.word_count for inf in inferences]),
        BERT_prec=mean_metric_ignoring_failures("BERT_prec"),
        BERT_rec=mean_metric_ignoring_failures("BERT_rec"),
        BERT_f1=mean_metric_ignoring_failures("BERT_f1"),
        ROUGE_1=mean_metric_ignoring_failures("ROUGE_1"),
        ROUGE_2=mean_metric_ignoring_failures("ROUGE_2"),
        ROUGE_L=mean_metric_ignoring_failures("ROUGE_L"),
        BLEU=mean_metric_ignoring_failures("BLEU"),
        inf_to_gt_word_count=mean_metric_ignoring_failures("inf_to_gt_word_count"),
    )


def compute_score_distribution_plot(
    score: str,
    metrics: List[Union[TestSampleMetric, Inference]],
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
    x_metrics: List[Union[TestSampleMetric, Inference]],
    y_metrics: List[Union[TestSampleMetric, Inference]],
    binning_info: Optional[Tuple[float, float, float]] = None,  # start, end, num
    x_logarithmic: bool = False,
    y_logarithmic: bool = False,
) -> CurvePlot:
    y_values = [getattr(m, y_metric) for m in y_metrics]
    x_values = [getattr(m, x_metric) for m in x_metrics]
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


def compute_plots(metrics: List[TestSampleMetric], test_samples: List[TestSample], infs: List[Inference]) -> List[Plot]:
    return [
        compute_score_distribution_plot("inf_to_gt_word_count", metrics, (0, 5.0, 51)),
        compute_score_distribution_plot("BERT_f1", metrics, (0, 1, 101)),
        compute_score_distribution_plot("ROUGE_L", metrics, (0, 1, 101)),
        compute_score_distribution_plot("BLEU", metrics, (-8, -0.5, 21), True),
        compute_score_distribution_plot("cost", infs, (-14, -3, 89), True),
        compute_metric_vs_metric_plot("word_count", "BERT_f1", test_samples, metrics, (50, 2000, 31)),
        compute_metric_vs_metric_plot("BERT_rec", "BERT_prec", metrics, metrics, (0.7, 1.0, 101)),
    ]


def compute_test_suite_metrics(
    test_samples: List[TestSample],
    inferences: List[Inference],
    metrics: List[Tuple[TestCase, TestCaseMetric]],
) -> TestSuiteMetric:
    num_failures = sum([inf.is_failure for inf in inferences])
    return TestSuiteMetric(
        num_articles=len(test_samples),
        num_failures=num_failures,
        failure_rate=num_failures / len(test_samples),
        total_cost=np.sum([inf.cost for inf in inferences]),
        variance_BERT_f1=np.var([m.BERT_f1 for _, m in metrics]),
        variance_BLEU=np.var([m.BLEU for _, m in metrics]),
        variance_ROUGE_L=np.var([m.ROUGE_L for _, m in metrics]),
    )


def evaluate_text_summarization(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
) -> EvaluationResults:
    print("computing test sample metrics...")
    test_sample_metrics = [compute_test_sample_metrics(gt, inf) for gt, inf in zip(ground_truths, inferences)]

    all_test_case_metrics: List[Tuple[TestCase, TestCaseMetric]] = []
    all_test_case_plots: List[Tuple[TestCase, List[Plot]]] = []
    for test_case, tc_test_samples, tc_gts, tc_infs, tc_ts_metrics in test_cases.iter(
        test_samples,
        ground_truths,
        inferences,
        test_sample_metrics,
    ):
        print(f"computing aggregate metrics for test case '{test_case.name}'...")
        test_case_metrics = compute_aggregate_metrics(tc_ts_metrics, tc_test_samples, tc_gts, tc_infs)
        all_test_case_metrics.append((test_case, test_case_metrics))

        print(f"computing plots for test case '{test_case.name}'...")
        test_case_plots = compute_plots(tc_ts_metrics, tc_test_samples, tc_infs)
        all_test_case_plots.append((test_case, test_case_plots))

    return EvaluationResults(
        metrics_test_sample=list(zip(test_samples, test_sample_metrics)),
        metrics_test_case=all_test_case_metrics,
        metrics_test_suite=compute_test_suite_metrics(test_samples, inferences, all_test_case_metrics),
        plots_test_case=all_test_case_plots,
    )
