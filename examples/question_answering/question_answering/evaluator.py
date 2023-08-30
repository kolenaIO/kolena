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
import pandas as pd
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
from question_answering.workflow import TestSuite
from question_answering.workflow import TestSuiteMetrics
from question_answering.workflow import ThresholdConfiguration

from kolena.workflow import Evaluator
from kolena.workflow import Plot

bertscore = evaluate.load("bertscore")
rouge = evaluate.load("rouge")


class QuestionAnswerEvaluator(Evaluator):
    """
    The `QuestionAnswerEvaluator` transforms inferences into metrics for the question answer workflow.

    For additional functionality, see the associated [base class documentation][kolena.workflow.evaluator.Evaluator].
    """

    test_samples_by_test_case: Dict[str, Dict[str, List[int]]] = {}  # test_case -> data_id -> turns
    """The cache for all test samples by test case."""

    precomputed_metrics: Dict[str, Dict[int, Dict[str, float]]] = {}  # data_id -> turn -> dict of metrics
    """The cache for all precomputed metrics."""

    def populate_cache(self, file_path: str) -> None:
        df = pd.read_csv(file_path)
        metrics = ["BERT_prec", "BERT_rec", "BERT_f1", "ROUGE_1", "ROUGE_2", "ROUGE_L"]

        for _, row in df.iterrows():
            data_id = row["data_id"]
            turn = row["turn"]
            values_dict = {col: round(row[col], 6) for col in metrics}

            self.precomputed_metrics.setdefault(data_id, {})[turn] = values_dict

    def compute_metrics(self, norm_gt_answer: str, norm_inf_answer: str) -> Dict[str, float]:
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

    def compute_test_sample_metric(
        self,
        ts: TestSample,
        gt: GroundTruth,
        inf: Inference,
        configuration: ThresholdConfiguration,
        is_cached: bool,
    ):
        if is_cached:
            results = self.precomputed_metrics[ts.data_id][ts.turn]
        else:
            results = self.compute_metrics(gt.clean_answer, inf.clean_answer)

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

    def compute_test_sample_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        configuration: Optional[ThresholdConfiguration] = None,
    ) -> List[Tuple[TestSample, TestSampleMetrics]]:
        assert configuration is not None, "must specify configuration"
        self.precomputed_metrics = {}

        for ts, _, _ in inferences:
            self.test_samples_by_test_case.setdefault(test_case.name, {}).setdefault(ts.data_id, []).append(ts.turn)

        # use precomputed metrics if possible
        first_inference_src = inferences[0][2].source
        is_cached = first_inference_src.startswith("s3://")
        if is_cached:
            self.populate_cache(first_inference_src.replace("results", "metrics"))

        return [
            (ts, self.compute_test_sample_metric(ts, gt, inf, configuration, is_cached)) for ts, gt, inf in inferences
        ]

    def compute_test_case_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        metrics: List[TestSampleMetrics],
        configuration: Optional[ThresholdConfiguration] = None,
    ) -> TestCaseMetrics:
        assert configuration is not None, "must specify configuration"

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
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        metrics: List[TestSampleMetrics],
        configuration: Optional[ThresholdConfiguration] = None,
    ) -> Optional[List[Plot]]:
        assert configuration is not None, "must specify configuration"
        metrics_of_interest = ["BERT_f1", "ROUGE_1", "MEAN_METRIC"]
        metric_values = [mean_metric(metric, metrics) for metric in metrics_of_interest]
        infs = [inf for _, _, inf in inferences]
        test_samples = [ts for ts, _, _ in inferences]

        plots: List[Plot] = [
            compute_metric_bar_plot(test_case.name, metrics_of_interest, metric_values),
            compute_score_distribution_plot("BERT_f1", metrics, (0, 1, 101)),
            compute_score_distribution_plot("ROUGE_1", metrics, (0, 1, 101)),
            compute_score_distribution_plot("ROUGE_L", metrics, (0, 1, 101)),
            compute_score_distribution_plot("MEAN_METRIC", metrics, (0, 1, 101)),
            compute_metric_vs_metric_plot("wc_answer", "BERT_f1", infs, metrics, (0, 50, 26)),
            compute_metric_vs_metric_plot("wc_answer", "ROUGE_1", infs, metrics, (0, 50, 26)),
            compute_metric_vs_metric_plot("turn", "ROUGE_1", test_samples, metrics, (0, 20, 26)),
            compute_metric_vs_metric_plot("turn", "BERT_f1", test_samples, metrics, (0, 20, 26)),
        ]

        return [plot for plot in plots if plot is not None]

    def compute_test_suite_metrics(
        self,
        test_suite: TestSuite,
        metrics: List[Tuple[TestCase, TestCaseMetrics]],
        configuration: Optional[ThresholdConfiguration] = None,
    ) -> TestSuiteMetrics:
        n_correct = sum([tcm.n_correct for _, tcm in metrics])
        test_samples = n_correct + sum([tcm.n_incorrect for _, tcm in metrics])
        unique_stories = {a for test_case_data in self.test_samples_by_test_case.values() for a in test_case_data}

        return TestSuiteMetrics(
            n_stories=len(unique_stories),
            n_questions=test_samples,
            n_correct=n_correct,
            overall_accuracy=round(n_correct / test_samples, 3),
        )
