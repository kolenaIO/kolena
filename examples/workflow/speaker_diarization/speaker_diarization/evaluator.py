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
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from jiwer import cer
from jiwer import wer
from pyannote.metrics.detection import DetectionAccuracy
from pyannote.metrics.detection import DetectionPrecision
from pyannote.metrics.detection import DetectionRecall
from pyannote.metrics.diarization import DiarizationCoverage
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.diarization import DiarizationPurity
from pyannote.metrics.diarization import JaccardErrorRate
from pyannote.metrics.identification import IdentificationErrorRate
from pyannote.metrics.identification import IdentificationPrecision
from pyannote.metrics.identification import IdentificationRecall
from speaker_diarization.utils import generate_annotation
from speaker_diarization.utils import generate_identification_error
from speaker_diarization.utils import generate_missed_speech_error
from speaker_diarization.utils import preprocess_text
from speaker_diarization.workflow import GroundTruth
from speaker_diarization.workflow import Inference
from speaker_diarization.workflow import TestCase
from speaker_diarization.workflow import TestCaseMetric
from speaker_diarization.workflow import TestSample
from speaker_diarization.workflow import TestSampleMetric
from speaker_diarization.workflow import TestSuiteMetric

from kolena.workflow import AxisConfig
from kolena.workflow import Curve
from kolena.workflow import CurvePlot
from kolena.workflow import Histogram
from kolena.workflow import Plot
from kolena.workflow.evaluator_function import EvaluationResults
from kolena.workflow.evaluator_function import TestCases


def compute_test_sample_metrics(gt: GroundTruth, inf: Inference) -> TestSampleMetric:
    reference = generate_annotation(gt.transcription)
    inference = generate_annotation(inf.transcription)

    gt_text = preprocess_text(gt.transcription)
    inf_text = preprocess_text(inf.transcription)

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
        IdentificationError=generate_identification_error(gt, inf),
        MissedSpeechError=generate_missed_speech_error(gt, inf),
    )


def compute_aggregate_metrics(
    test_samples_metrics: List[TestSampleMetric],
    test_samples: List[TestSample],
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
    metrics: List[TestSampleMetric],
    binning_info: Optional[Tuple[float, float, float]] = None,  # start, end, num
    logarithmic_y: bool = False,
    logarithmic_x: bool = False,
) -> Histogram:
    scores = [getattr(m, score) for m in metrics]
    if logarithmic_x:
        bins = np.logspace(*binning_info, base=2)  # type: ignore
    else:
        bins = np.linspace(*binning_info)  # type: ignore

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
    x_metrics: List[TestSample],
    y_metrics: List[TestSampleMetric],
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
        bins = list(np.logspace(*binning_info, base=2))  # type: ignore
    else:
        bins = list(np.linspace(*binning_info))  # type: ignore

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
    test_case_metrics: List[TestSampleMetric],
    test_samples: List[TestSample],
) -> List[Plot]:
    return [
        compute_score_distribution_plot("DiarizationErrorRate", test_case_metrics, (0, 1, 20)),
        compute_score_distribution_plot("DetectionAccuracy", test_case_metrics, (0, 1, 20)),
        compute_score_distribution_plot("IdentificationErrorRate", test_case_metrics, (0, 1, 20)),
        compute_metric_vs_metric_plot(
            "Average_Amplitude",
            "DiarizationErrorRate",
            test_samples,
            test_case_metrics,
            (  # type: ignore
                min([ts.metadata["Average_Amplitude"] for ts in test_samples]),  # type: ignore
                max([ts.metadata["Average_Amplitude"] for ts in test_samples]),  # type: ignore
                15,
            ),
            metadata=True,
        ),
    ]


def compute_test_suite_metrics(
    inferences: List[Inference],
) -> TestSuiteMetric:
    return TestSuiteMetric(
        Diarizations=len(inferences),
    )


def evaluate_speaker_diarization(
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
        test_case_metrics = compute_aggregate_metrics(tc_ts_metrics, tc_test_samples)
        all_test_case_metrics.append((test_case, test_case_metrics))

        print(f"computing plots for test case '{test_case.name}'...")
        test_case_plots = compute_test_case_plots(tc_ts_metrics, tc_test_samples)
        all_test_case_plots.append((test_case, test_case_plots))

    return EvaluationResults(
        metrics_test_sample=list(zip(test_samples, test_sample_metrics)),
        metrics_test_case=all_test_case_metrics,  # type: ignore
        metrics_test_suite=compute_test_suite_metrics(inferences),
        plots_test_case=all_test_case_plots,
    )
