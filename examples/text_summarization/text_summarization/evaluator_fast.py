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
from typing import Tuple

import pandas as pd
from text_summarization.evaluator import compute_aggregate_metrics
from text_summarization.evaluator import compute_plots
from text_summarization.evaluator import compute_test_suite_metrics
from text_summarization.workflow import GroundTruth
from text_summarization.workflow import Inference
from text_summarization.workflow import TestCase
from text_summarization.workflow import TestCaseMetric
from text_summarization.workflow import TestSample
from text_summarization.workflow import TestSampleMetric

from kolena.workflow import Plot
from kolena.workflow.evaluator_function import EvaluationResults
from kolena.workflow.evaluator_function import TestCases


def compute_test_sample_metrics(ts: TestSample, gt: GroundTruth, inf: Inference, df: pd.DataFrame) -> TestSampleMetric:
    record = df.loc[df["article_id"] == ts.id]
    return TestSampleMetric(
        BERT_prec=float(record["BERT_prec"].values[0]) if not inf.is_failure else 0.0,
        BERT_rec=float(record["BERT_rec"].values[0]) if not inf.is_failure else 0.0,
        BERT_f1=float(record["BERT_f1"].values[0]) if not inf.is_failure else 0.0,
        ROUGE_1=float(record["ROUGE_1"].values[0]) if not inf.is_failure else 0.0,
        ROUGE_2=float(record["ROUGE_2"].values[0]) if not inf.is_failure else 0.0,
        ROUGE_L=float(record["ROUGE_L"].values[0]) if not inf.is_failure else 0.0,
        BLEU=float(record["BLEU"].values[0]) if not inf.is_failure else 0.0,
        inf_to_gt_word_count=float(inf.word_count) / gt.word_count,
    )


def evaluate_text_summarization_fast(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
) -> EvaluationResults:
    print("computing test sample metrics...")
    df = pd.read_csv(inferences[0].source)
    # sanitize column names to use underscores instead of spaces, dots, dashes, or slashes
    df.columns = df.columns.str.replace(r"(\s|\.|\/|\-)+", "_", regex=True)

    # Compute test sample metrics
    test_sample_metrics: List[TestSampleMetric] = [
        compute_test_sample_metrics(ts, gt, inf, df) for ts, gt, inf in zip(test_samples, ground_truths, inferences)
    ]

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
