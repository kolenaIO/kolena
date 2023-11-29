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
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import evaluate
from rag_qa.utils import compute_metric_bar_plot
from rag_qa.utils import compute_score_distribution_plot
from rag_qa.utils import mean_metric
from rag_qa.workflow import Configuration
from rag_qa.workflow import TestCase

from kolena.workflow import Inference
from kolena.workflow import MetricsTestCase
from kolena.workflow import MetricsTestSample
from kolena.workflow import MetricsTestSuite
from kolena.workflow import Plot
from kolena.workflow import Text

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


def compute_test_sample_metrics(gt: str, inf: str, configuration: Configuration):
    results = compute_metrics(gt, inf)
    custom_metric = round((results["BERT_f1"] + results["ROUGE_1"]) / 2, 3)
    return MetricsTestSample(
        BERT_prec=results["BERT_prec"],
        BERT_rec=results["BERT_rec"],
        BERT_f1=results["BERT_f1"],
        MEAN_METRIC=custom_metric,
        ROUGE_1=results["ROUGE_1"],
        ROUGE_2=results["ROUGE_2"],
        ROUGE_L=results["ROUGE_L"],
    )


def compute_test_case_metrics(
    metrics: List[MetricsTestSample],
) -> MetricsTestCase:
    return MetricsTestCase(
        BERT_prec=mean_metric("BERT_prec", metrics),
        BERT_rec=mean_metric("BERT_rec", metrics),
        BERT_f1=mean_metric("BERT_f1", metrics),
        MEAN_METRIC=mean_metric("MEAN_METRIC", metrics),
        ROUGE_1=mean_metric("ROUGE_1", metrics),
        ROUGE_2=mean_metric("ROUGE_2", metrics),
        ROUGE_L=mean_metric("ROUGE_L", metrics),
    )


def compute_test_case_plots(
    test_samples: List[Text],
    inferences: List[Inference],
    metrics: List[MetricsTestSample],
) -> Optional[List[Plot]]:
    metrics_of_interest = ["BERT_f1", "ROUGE_1", "MEAN_METRIC"]
    metric_values = [mean_metric(metric, metrics) for metric in metrics_of_interest]

    plots: List[Plot] = [
        compute_metric_bar_plot(metrics_of_interest, metric_values),
        compute_score_distribution_plot("BERT_f1", metrics, (0, 1, 101)),
        compute_score_distribution_plot("ROUGE_1", metrics, (0, 1, 101)),
        compute_score_distribution_plot("ROUGE_L", metrics, (0, 1, 101)),
        compute_score_distribution_plot("MEAN_METRIC", metrics, (0, 1, 101)),
    ]

    return [plot for plot in plots if plot is not None]


def compute_test_suite_metrics(
    test_samples: List[Text],
    test_case_metrics: List[Tuple[TestCase, MetricsTestCase]],
) -> MetricsTestSuite:
    return MetricsTestSuite()
