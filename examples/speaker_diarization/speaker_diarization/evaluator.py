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

import re
import numpy as np
from jiwer import cer, wer
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
from kolena.workflow.annotation import ClassificationLabel
from kolena.workflow.evaluator_function import EvaluationResults
from kolena.workflow.evaluator_function import TestCases

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate, DiarizationPurity, DiarizationCoverage
from pyannote.metrics.detection import DetectionAccuracy, DetectionPrecision, DetectionRecall
from pyannote.metrics.identification import IdentificationErrorRate, IdentificationPrecision, IdentificationRecall

from utils import generate_tp, generate_fp, inv

def compute_test_sample_metrics(gt: GroundTruth, inf: Inference) -> TestSampleMetric:
    reference = Annotation()
    for row in gt.transcription:
        reference[Segment(row.start, row.end)] = row.group
    inference = Annotation()
    for row in inf.transcription:
        inference[Segment(row.start, row.end)] = row.group
    
    gt_text = " ".join([row.label for row in gt.transcription])
    inf_text= " ".join([row.label for row in inf.transcription])
    gt_text = re.sub(r"[^\w\s]", "", gt_text.lower())
    inf_text = re.sub(r"[^\w\s]", "", inf_text.lower())


    return TestSampleMetric(
        DiarizationErrorRate=DiarizationErrorRate()(reference, inference),
        JaccardErrorRate=JaccardErrorRate()(reference, inference),
        DiarizationPurity=DiarizationPurity()(reference, inference),
        DiarizationCoverage=DiarizationCoverage()(reference, inference),

        DetectionAccuracy=DetectionAccuracy()(reference, inference),
        DetectionPrecision=DetectionPrecision()(reference, inference),
        DetectionRecall=DetectionRecall()(reference, inference),

        IdentificationErrorRate=IdentificationErrorRate()(reference, inference),
        IdentificationPrecision=IdentificationPrecision()(reference, inference),
        IdentificationRecall=IdentificationRecall()(reference, inference),

        WordErrorRate=wer(gt_text, inf_text),
        CharacterErrorRate=cer(gt_text, inf_text),

        IdentificationError=generate_fp(gt, inf, identification=True),
        MissedSpeechError=generate_tp(gt, inv(inf, gt), identification=False),
    )


def compute_aggregate_metrics(
    test_samples_metrics: List[TestSampleMetric],
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
) -> TestCaseMetric:
    n_samples = len(test_samples)
    return TestCaseMetric(
        DiarizationErrorRate=sum([metric.DiarizationErrorRate for metric in test_samples_metrics]) / n_samples,
        JaccardErrorRate=sum([metric.JaccardErrorRate for metric in test_samples_metrics]) / n_samples,
        DiarizationPurity=sum([metric.DiarizationPurity for metric in test_samples_metrics]) / n_samples,
        DiarizationCoverage=sum([metric.DiarizationCoverage for metric in test_samples_metrics]) / n_samples,

        DetectionAccuracy=sum([metric.DetectionAccuracy for metric in test_samples_metrics]) / n_samples,
        DetectionPrecision=sum([metric.DetectionPrecision for metric in test_samples_metrics]) / n_samples,
        DetectionRecall=sum([metric.DetectionRecall for metric in test_samples_metrics]) / n_samples,

        IdentificationErrorRate=sum([metric.IdentificationErrorRate for metric in test_samples_metrics]) / n_samples,
        IdentificationPrecision=sum([metric.IdentificationPrecision for metric in test_samples_metrics]) / n_samples,
        IdentificationRecall=sum([metric.IdentificationRecall for metric in test_samples_metrics]) / n_samples,
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
    return [
        compute_score_distribution_plot("DiarizationErrorRate", test_case_metrics, (0, 1, 20)),
        compute_score_distribution_plot("JaccardErrorRate", test_case_metrics, (0, 1, 20)),
        compute_score_distribution_plot("DetectionAccuracy", test_case_metrics, (0, 1, 20)),
        compute_score_distribution_plot("IdentificationErrorRate", test_case_metrics, (0, 1, 20)),
    ]


def compute_test_suite_metrics(
    test_samples: List[TestSample],
    inferences: List[Inference],
    metrics: List[Tuple[TestCase, TestCaseMetric]],
) -> TestSuiteMetric:

    return TestSuiteMetric(
        Diarizations=len(inferences)
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