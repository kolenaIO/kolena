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

import langcodes
import langid
import numpy as np
from jiwer import cer
from jiwer import process_words
from numwords_to_nums.numwords_to_nums import NumWordsToNum
from workflow import GroundTruth
from workflow import Inference
from workflow import TestCase
from workflow import TestCaseMetric
from workflow import TestSample
from workflow import TestSampleMetric
from workflow import TestSuiteMetric

from kolena.workflow import AxisConfig
from kolena.workflow import Curve
from kolena.workflow import CurvePlot
from kolena.workflow import Histogram
from kolena.workflow import Plot
from kolena.workflow.evaluator_function import EvaluationResults
from kolena.workflow.evaluator_function import TestCases
from kolena.workflow.annotation import ClassificationLabel

from utils import generate_diff_word_level, preprocess_transcription


def compute_test_sample_metrics(gt: GroundTruth, inf: Inference) -> TestSampleMetric:
    gt = preprocess_transcription(gt)
    inf = preprocess_transcription(inf)

    word_fp, (fn_count, fp_count, ins_count, del_count, sub_count, sub_list, ins_list, del_list) = generate_diff_word_level(gt, inf, mode="fp")
    word_fn, _ = generate_diff_word_level(gt, inf, mode="fn")

    wer_metrics = process_words(gt, inf)
    word_errors = ins_count + sub_count + del_count
    word_error_rate = wer_metrics.wer
    match_error_rate = wer_metrics.mer
    word_information_lost = wer_metrics.wil
    word_information_preserved = wer_metrics.wip
    character_error_rate = cer(gt, inf)

    language = langcodes.Language.get(langid.classify(inf)[0]).display_name()

    return TestSampleMetric(
        WordErrors=word_errors,
        WordErrorRate=word_error_rate,
        MatchErrorRate=match_error_rate,
        WordInformationLost=word_information_lost,
        WordInformationPreserved=word_information_preserved,
        CharacterErrorRate=character_error_rate,
        FalsePositiveText=word_fp,
        FalseNegativeText=word_fn,
        FNCount=fn_count,
        FPCount=fp_count,
        InsertionCount=ins_count,
        DeletionCount=del_count,
        SubstitutionCount=sub_count,
        Language=language,

        Substitutions = [ClassificationLabel(item) for sublist in sub_list for item in (sublist if isinstance(sublist, list) else [sublist])],
        Insertions = [ClassificationLabel(item) for sublist in ins_list for item in (sublist if isinstance(sublist, list) else [sublist])],
        Deletions = [ClassificationLabel(item) for sublist in del_list for item in (sublist if isinstance(sublist, list) else [sublist])],
    )


def compute_aggregate_metrics(
    test_samples_metrics: List[TestSampleMetric],
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
) -> TestCaseMetric:
    n_samples = 0
    n_failures = 0
    sum_word_errors = 0
    sum_word_error_rate = 0
    sum_match_error_rate = 0
    sum_word_information_lost = 0
    sum_word_information_preserved = 0
    sum_character_error_rate = 0
    sum_wc_gt = 0
    sum_wc_inf = 0

    for metric in test_samples_metrics:
        if metric.WordErrors != 0:
            n_failures += 1
        sum_word_errors += metric.WordErrors
        sum_word_error_rate += metric.WordErrorRate
        sum_match_error_rate += metric.MatchErrorRate
        sum_word_information_lost += metric.WordInformationLost
        sum_word_information_preserved += metric.WordInformationPreserved
        sum_character_error_rate += metric.CharacterErrorRate
        n_samples += 1

    for gt in ground_truths:
        sum_wc_gt += len(gt.transcription.label.split(" "))
    for inf in inferences:
        sum_wc_inf += len(inf.transcription.label.split(" "))

    return TestCaseMetric(
        FailCount=n_failures,
        FailRate=n_failures / n_samples,
        AvgWordErrors=sum_word_errors / n_samples,
        WordErrorRate=sum_word_error_rate / n_samples,
        MatchErrorRate=sum_match_error_rate / n_samples,
        WordInfoLost=sum_word_information_lost / n_samples,
        WordInfoPreserved=sum_word_information_preserved / n_samples,
        CharacterErrorRate=sum_character_error_rate / n_samples,
        AvgGTWordCount=sum_wc_gt / n_samples,
        AvgInfWordCount=sum_wc_inf / n_samples,
    )


def compute_score_distribution_plot(
    score: str,
    metrics: List[Union[TestSampleMetric, Inference]],
    binning_info: Optional[Tuple[float, float, float]] = None,  # start, end, num
    logarithmic_y: bool = False,
    logarithmic_x: bool = False,
) -> Histogram:
    scores = [getattr(m, score) for m in metrics]
    if logarithmic_x:
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
        y_config=AxisConfig(type="log") if logarithmic_y else None,
        x_config=AxisConfig(type="log") if logarithmic_x else None,
    )


def compute_metric_vs_metric_plot(
    x_metric: str,
    y_metric: str,
    x_metrics: List[Union[TestSampleMetric, Inference]],
    y_metrics: List[Union[TestSampleMetric, Inference]],
    binning_info: Optional[Tuple[float, float, float]] = None,  # start, end, num
    x_logarithmic: bool = False,
    y_logarithmic: bool = False,
    metadata: bool = False,
) -> CurvePlot:
    y_values = [getattr(m, y_metric) for m in y_metrics]
    if metadata:
        x_values = [m.metadata[x_metric] for m in x_metrics]
    else:
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


def compute_test_case_plots(
    complete_metrics: List[TestSampleMetric],
    test_case_metrics: List[TestSampleMetric],
    test_samples: List[TestSample],
) -> List[Plot]:
    max_word_error = max([metric.WordErrors for metric in complete_metrics])

    return [
        compute_score_distribution_plot(
            "WordErrors",
            test_case_metrics,
            (0, max_word_error + 1, int(max_word_error)),
            logarithmic_y=True,
        ),
        compute_score_distribution_plot("WordErrorRate", test_case_metrics, (0, 1, 51), logarithmic_y=True),
        compute_score_distribution_plot("CharacterErrorRate", test_case_metrics, (0, 1, 51), logarithmic_y=True),
        compute_score_distribution_plot("WordInformationLost", test_case_metrics, (0, 1, 51), logarithmic_y=True),
        compute_metric_vs_metric_plot(
            "duration_seconds",
            "WordErrorRate",
            test_samples,
            test_case_metrics,
            (0, 35, 7),
            metadata=True,
        ),
        compute_metric_vs_metric_plot(
            "duration_seconds",
            "CharacterErrorRate",
            test_samples,
            test_case_metrics,
            (0, 35, 7),
            metadata=True,
        ),
        compute_metric_vs_metric_plot(
            "tempo",
            "WordErrorRate",
            test_samples,
            test_case_metrics,
            (0, 6, 14),
            metadata=True,
        ),
        compute_metric_vs_metric_plot(
            "tempo",
            "CharacterErrorRate",
            test_samples,
            test_case_metrics,
            (0, 6, 14),
            metadata=True,
        ),
    ]


def compute_test_suite_metrics(
    test_samples: List[TestSample],
    inferences: List[Inference],
    metrics: List[Tuple[TestCase, TestCaseMetric]],
) -> TestSuiteMetric:
    transcriptions = 0
    failures = 0
    failurerate = 0
    for metric in metrics:
        if "complete" in metric[0].name:
            transcriptions = len(test_samples)
            failures = metric[1].FailCount
            failurerate = metric[1].FailRate
            break

    return TestSuiteMetric(
        Transcriptions=transcriptions,
        Failures=failures,
        FailureRate=failurerate,
    )


def evaluate_audio_recognition(
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
        test_case_plots = compute_test_case_plots(test_sample_metrics, tc_ts_metrics, tc_test_samples)
        all_test_case_plots.append((test_case, test_case_plots))

    return EvaluationResults(
        metrics_test_sample=list(zip(test_samples, test_sample_metrics)),
        metrics_test_case=all_test_case_metrics,
        metrics_test_suite=compute_test_suite_metrics(test_samples, inferences, all_test_case_metrics),
        plots_test_case=all_test_case_plots,
    )
