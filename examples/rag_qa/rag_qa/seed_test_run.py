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
from argparse import ArgumentParser
from argparse import Namespace
from collections import OrderedDict
from typing import Callable
from typing import List
from typing import Tuple

from rag_qa.constants import HALU_DIALOG
from rag_qa.constants import HALU_QA
from rag_qa.constants import HALU_SUMMARIZATION
from rag_qa.constants import SQUAD2_DEV
from rag_qa.data_loader import HALU_MODELS
from rag_qa.data_loader import load_halu_dialog_results
from rag_qa.data_loader import load_halu_qa_results
from rag_qa.data_loader import load_halu_summarization_results
from rag_qa.data_loader import load_squad2_dev_results
from rag_qa.data_loader import SQUAD_MODELS
from rag_qa.evaluator import compute_test_case_metrics
from rag_qa.evaluator import compute_test_case_plots
from rag_qa.evaluator import compute_test_sample_metrics
from rag_qa.evaluator import compute_test_suite_metrics
from rag_qa.utils import normalize_string
from rag_qa.workflow import Configuration
from rag_qa.workflow import Inference
from rag_qa.workflow import Model
from rag_qa.workflow import TestCase
from rag_qa.workflow import TestSuite
from tqdm import tqdm

import kolena
from kolena.workflow import BasicEvaluatorFunction
from kolena.workflow import EvaluationResults
from kolena.workflow import GroundTruth
from kolena.workflow import MetricsTestCase
from kolena.workflow import Plot
from kolena.workflow import test
from kolena.workflow import TestCases
from kolena.workflow import Text

results_loader = OrderedDict(
    [
        (SQUAD2_DEV, load_squad2_dev_results),
        (HALU_QA, load_halu_qa_results),
        (HALU_DIALOG, load_halu_dialog_results),
        (HALU_SUMMARIZATION, load_halu_summarization_results),
    ],
)

key_columns = {
    SQUAD2_DEV: ["id"],
    HALU_QA: ["text", "question"],
    HALU_DIALOG: ["text", "dialogue_history"],
    # TODO: does not work yet due to non-trivial matching between test-sample and inference
    HALU_SUMMARIZATION: ["text"],
}

answer_getters = {
    SQUAD2_DEV: lambda gt, inf: (gt.answers[0] if gt.answers else "", inf.answer),
    HALU_QA: lambda gt, inf: (gt.right_answer, inf.answer),
    HALU_DIALOG: lambda gt, inf: (gt.right_response, inf.answer),
    HALU_SUMMARIZATION: lambda gt, inf: (gt.right_summary, inf.answer),
}


def get_evaluator(getter: Callable[[GroundTruth, Inference], Tuple[str, str]]) -> BasicEvaluatorFunction:
    def evaluator(
        test_samples: List[Text],
        ground_truths: List[GroundTruth],
        inferences: List[Inference],
        test_cases: TestCases,
        configuration: Configuration,
    ) -> EvaluationResults:
        # compute sample-level metrics for each sample
        test_sample_metrics = []
        for gt, inf in tqdm(zip(ground_truths, inferences), total=len(ground_truths)):
            gt_val, inf_val = getter(gt, inf)
            test_sample_metrics.append(
                compute_test_sample_metrics(normalize_string(gt_val), normalize_string(inf_val), configuration),
            )

        # compute aggregate metrics across all test cases using `test_cases.iter(...)`
        all_test_case_metrics: List[Tuple[TestCase, MetricsTestCase]] = []
        all_test_case_plots: List[Tuple[TestCase, List[Plot]]] = []
        for test_case, ts, gt, inf, tsm in test_cases.iter(
            test_samples,
            ground_truths,
            inferences,
            test_sample_metrics,
        ):
            all_test_case_metrics.append((test_case, compute_test_case_metrics(tsm)))
            all_test_case_plots.append((test_case, compute_test_case_plots(ts, inf, tsm)))

        test_suite_metrics = compute_test_suite_metrics(test_samples, all_test_case_metrics)

        # if desired, compute and add `plots_test_case` and `metrics_test_suite` to this `EvaluationResults`
        return EvaluationResults(
            metrics_test_sample=list(zip(test_samples, test_sample_metrics)),
            metrics_test_case=all_test_case_metrics,
            plots_test_case=all_test_case_plots,
            metrics_test_suite=test_suite_metrics,
        )

    return evaluator


def main(args: Namespace) -> None:
    kolena.initialize(verbose=True)

    benchmark = args.benchmark
    test_suite_name = args.test_suite or benchmark
    df, config = results_loader[args.benchmark](args.model)
    keys = key_columns[benchmark]
    df = df.set_index(keys)

    # define a function that generates an inference from a test sample
    def infer(test_sample: Text) -> Inference:
        target = [getattr(test_sample, k) for k in keys]
        result = df.loc[target].to_dict(orient="records")[0]
        return Inference(**result)

    model = Model(args.model, infer=infer)
    test_suite = TestSuite.load(test_suite_name)
    evaluator = get_evaluator(answer_getters[benchmark])
    test(model, test_suite, evaluator, [Configuration(**config)], reset=True)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--benchmark",
        choices=[SQUAD2_DEV, HALU_QA, HALU_DIALOG, HALU_SUMMARIZATION],
        required=True,
        help="Name of the benchmark to test.",
    )
    ap.add_argument("--model", choices=SQUAD_MODELS + HALU_MODELS, help="Name of the model to test.")
    ap.add_argument("--test_suite", help="Name of the test suite to test.")

    main(ap.parse_args())
