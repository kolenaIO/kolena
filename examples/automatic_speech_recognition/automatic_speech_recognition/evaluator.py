from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import re
import difflib
from jiwer import cer, process_words
import numpy as np
import langid
import langcodes

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

def compute_test_sample_metrics(gt: GroundTruth, inf: Inference) -> TestSampleMetric:

    def generate_diff_word_level(reference: str, candidate: str, mode: str):
        matcher = difflib.SequenceMatcher(None, reference.split(), candidate.split())
        fp_count = 0
        fn_count = 0
        
        output = []
        for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
            if opcode == 'equal':
                output.append(" ".join(matcher.a[a0:a1]))
                
            elif opcode == 'insert':
                fn_count += 1
                if mode == "fn":
                    output.append(f"<fn>" + " ".join(matcher.b[b0:b1]) + f"</fn>")
                else:
                    output.append(" ".join(matcher.b[b0:b1]))

            elif opcode == 'delete':
                fp_count += 1
                if mode == "fp":
                    output.append(f"<fp>" + " ".join(matcher.a[a0:a1]) + f"</fp>")
                else:
                    output.append(" ".join(matcher.a[a0:a1]))

            elif opcode == 'replace':
                fn_count += 1
                fp_count += 1
                if mode == "fp":
                    output.append(f"<fp>" + " ".join(matcher.b[b0:b1]) + f"</fp>")
                elif mode == "fn":
                    output.append(f"<fn>" + " ".join(matcher.a[a0:a1]) + f"</fn>")
                else:
                    output.append(" ".join(matcher.b[b0:b1]))
        
        return " ".join(output), fn_count, fp_count

    gt = re.sub(r'[^\w\s]', '', gt.transcription.label.lower())
    inf = re.sub(r'[^\w\s]', '', inf.transcription.label.lower())

    wer_metrics = process_words(gt, inf)
    word_errors = wer_metrics.substitutions + wer_metrics.deletions + wer_metrics.insertions
    word_error_rate = wer_metrics.wer
    match_error_rate = wer_metrics.mer
    word_information_lost = wer_metrics.wil
    word_information_preserved = wer_metrics.wip
    character_error_rate = cer(gt, inf)

    word_fp, fn_count, fp_count = generate_diff_word_level(gt, inf, mode="fp")
    word_fn, _, _ = generate_diff_word_level(gt, inf, mode="fn")

    language = langcodes.Language.get(langid.classify(inf)[0]).display_name()

    return TestSampleMetric(
        word_errors=word_errors,
        word_error_rate=word_error_rate,
        match_error_rate=match_error_rate,
        word_information_lost=word_information_lost,
        word_information_preserved=word_information_preserved,
        character_error_rate=character_error_rate,
        word_fp=word_fp,
        word_fn=word_fn,
        fn_count=fn_count,
        fp_count=fp_count,
        language=language,
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
        if metric.word_errors != 0:
            n_failures += 1
        sum_word_errors += metric.word_errors
        sum_word_error_rate += metric.word_error_rate
        sum_match_error_rate += metric.match_error_rate
        sum_word_information_lost += metric.word_information_lost
        sum_word_information_preserved += metric.word_information_preserved
        sum_character_error_rate += metric.character_error_rate
        n_samples += 1
    
    for gt in ground_truths:
        sum_wc_gt += len(gt.transcription.label.split(' '))
    for inf in inferences:
        sum_wc_inf += len(inf.transcription.label.split(' '))

    return TestCaseMetric(
        n_failures=n_failures,
        failure_rate=n_failures / n_samples,
        avg_word_errors=sum_word_errors / n_samples,
        avg_word_error_rate=sum_word_error_rate / n_samples,
        avg_match_error_rate=sum_match_error_rate / n_samples,
        avg_word_information_lost=sum_word_information_lost / n_samples,
        avg_word_information_preserved=sum_word_information_preserved / n_samples,
        avg_character_error_rate=sum_character_error_rate / n_samples,
        avg_wc_gt=sum_wc_gt/n_samples,
        avg_wc_inf=sum_wc_inf/n_samples,
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


def compute_plots(metrics: List[TestSampleMetric], test_samples: List[TestSample], infs: List[Inference]) -> List[Plot]:
    max_word_error = 0
    for metric in metrics:
        if metric.word_errors > max_word_error:
            max_word_error = metric.word_errors

    return [
        compute_score_distribution_plot("word_errors", metrics, (0, max_word_error + 1, int(max_word_error)), logarithmic_y=True),
        compute_score_distribution_plot("word_error_rate", metrics, (0, 1, 51), logarithmic_y=True),
        compute_score_distribution_plot("character_error_rate", metrics, (0, 1, 51), logarithmic_y=True),
        compute_score_distribution_plot("word_information_lost", metrics, (0, 1, 51), logarithmic_y=True),

        compute_metric_vs_metric_plot("duration_seconds", "word_error_rate", test_samples, metrics, (0, 35, 7), metadata=True),
        compute_metric_vs_metric_plot("duration_seconds", "character_error_rate", test_samples, metrics, (0, 35, 7), metadata=True),
    ]


def compute_test_suite_metrics(
    test_samples: List[TestSample],
    inferences: List[Inference],
    metrics: List[Tuple[TestCase, TestCaseMetric]],
) -> TestSuiteMetric:
    
    n_samples = 0
    n_failures = 0

    # for _, metric in metrics:
    #     if metric.word_errors != 0:
    #         n_failures += 1
    #     n_samples += 1
    # FIX THIS LATER

    return TestSuiteMetric(
        num_transcriptions=0,
        num_failures=0,
        failure_rate=0,
        variance_word_error_rate=np.var([m.avg_word_error_rate for _, m in metrics]),
        variance_match_error_rate=np.var([m.avg_match_error_rate for _, m in metrics]),
        variance_word_information_lost=np.var([m.avg_word_information_lost for _, m in metrics]),
        variance_word_information_preserved=np.var([m.avg_word_information_lost for _, m in metrics]),
        variance_character_error_rate=np.var([m.avg_character_error_rate for _, m in metrics]),
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
        test_case_plots = compute_plots(tc_ts_metrics, tc_test_samples, tc_infs)
        all_test_case_plots.append((test_case, test_case_plots))

    return EvaluationResults(
        metrics_test_sample=list(zip(test_samples, test_sample_metrics)),
        metrics_test_case=all_test_case_metrics,
        metrics_test_suite=compute_test_suite_metrics(test_samples, inferences, all_test_case_metrics),
        plots_test_case=all_test_case_plots
    )