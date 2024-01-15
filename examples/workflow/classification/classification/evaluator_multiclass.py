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

import numpy as np
from classification.evaluator_base import BaseClassificationEvaluator
from classification.workflow import ClassMetricsPerTestCase
from classification.workflow import GroundTruth
from classification.workflow import Inference
from classification.workflow import TestCaseMetrics
from classification.workflow import TestSampleMetrics
from classification.workflow import ThresholdConfiguration

from kolena._experimental.classification.utils import compute_confusion_matrix
from kolena._experimental.classification.utils import compute_roc_curves
from kolena._utils import log
from kolena.metrics import accuracy as compute_accuracy
from kolena.metrics import f1_score as compute_f1_score
from kolena.metrics import precision as compute_precision
from kolena.metrics import recall as compute_recall
from kolena.workflow import Plot


class MulticlassClassificationEvaluator(BaseClassificationEvaluator):
    def compute_test_sample_metrics(
        self,
        ground_truth: GroundTruth,
        inference: Inference,
        threshold_configuration: ThresholdConfiguration,
    ) -> TestSampleMetrics:
        empty_metrics = TestSampleMetrics(
            classification=None,
            margin=None,
            threshold=None,
            is_correct=False,
        )

        if len(inference.inferences) == 0:
            return empty_metrics

        sorted_infs = sorted(inference.inferences, key=lambda x: x.score, reverse=True)
        predicted_match = sorted_infs[0]
        predicted_label, predicted_score = predicted_match.label, predicted_match.score

        if threshold_configuration.threshold is not None and predicted_score < threshold_configuration.threshold:
            return empty_metrics

        return TestSampleMetrics(
            classification=predicted_match,
            margin=predicted_score - sorted_infs[1].score if len(sorted_infs) >= 2 else None,
            is_correct=predicted_label == ground_truth.classification.label,
            threshold=threshold_configuration.threshold,
        )

    def compute_test_case_metrics(
        self,
        ground_truths: List[GroundTruth],
        metrics_test_samples: List[TestSampleMetrics],
    ) -> TestCaseMetrics:
        classification_pairs = [
            (gt.classification.label, tsm.classification.label if tsm.classification else None)
            for gt, tsm in zip(ground_truths, metrics_test_samples)
        ]
        n_images = len(classification_pairs)
        labels = {gt.classification.label for gt in ground_truths}
        class_level_metrics: List[ClassMetricsPerTestCase] = []
        for label in sorted(labels):
            n_tp = len([True for gt, inf in classification_pairs if gt == label and inf == label])
            n_fn = len([True for gt, inf in classification_pairs if gt == label and inf != label])
            n_fp = len([True for gt, inf in classification_pairs if gt != label and inf == label])
            n_tn = len([True for gt, inf in classification_pairs if gt != label and inf != label])

            class_level_metrics.append(
                ClassMetricsPerTestCase(
                    label=label,
                    nImages=n_tp + n_fn,
                    TP=n_tp,
                    FP=n_fp,
                    FN=n_fn,
                    TN=n_tn,
                    Accuracy=compute_accuracy(n_tp, n_fp, n_fn, n_tn),
                    Precision=compute_precision(n_tp, n_fp),
                    Recall=compute_recall(n_tp, n_fn),
                    F1=compute_f1_score(n_tp, n_fp, n_fn),
                    FPR=n_fp / (n_fp + n_tn) if n_fp + n_tn > 0 else 0,
                ),
            )

        n_correct = sum([mts.is_correct for mts in metrics_test_samples])
        return TestCaseMetrics(
            PerClass=class_level_metrics,
            n_labels=len(labels),
            n_correct=n_correct,
            n_incorrect=n_images - n_correct,
            Accuracy=n_correct / n_images,
            macro_Accuracy=np.mean([class_metric.Accuracy for class_metric in class_level_metrics]),
            macro_Precision=np.mean([class_metric.Precision for class_metric in class_level_metrics]),
            macro_Recall=np.mean([class_metric.Recall for class_metric in class_level_metrics]),
            macro_F1=np.mean([class_metric.F1 for class_metric in class_level_metrics]),
            macro_FPR=np.mean([class_metric.FPR for class_metric in class_level_metrics]),
        )

    def compute_test_case_plots(
        self,
        ground_truths: List[GroundTruth],
        inferences: List[Inference],
        metrics: List[TestSampleMetrics],
        gt_labels: List[str],
        confidence_range: Optional[Tuple[float, float, int]],
    ) -> List[Plot]:
        plots: List[Plot] = []

        if confidence_range:
            plots.extend(self._compute_test_case_confidence_histograms(metrics, confidence_range))
        else:
            log.warn("skipping test case confidence histograms: unsupported confidence range")

        plots.append(
            compute_roc_curves(
                [gt.classification for gt in ground_truths],
                [inf.inferences for inf in inferences],
                gt_labels,
            ),
        )
        plots.append(
            compute_confusion_matrix(
                [gt.classification.label for gt in ground_truths],
                [metric.classification.label if metric.classification is not None else "None" for metric in metrics],
            ),
        )
        plots = list(filter(lambda plot: plot is not None, plots))

        return plots
