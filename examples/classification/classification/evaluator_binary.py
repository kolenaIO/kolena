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

from classification.evaluator_base import BaseClassificationEvaluator
from classification.workflow import GroundTruth
from classification.workflow import Inference
from classification.workflow import TestCaseMetricsSingleClass
from classification.workflow import TestSampleMetricsSingleClass
from classification.workflow import ThresholdConfiguration

from kolena._experimental.classification.utils import compute_confusion_matrix
from kolena._experimental.classification.utils import compute_roc_curves
from kolena._experimental.classification.utils import compute_threshold_curves
from kolena._utils import log
from kolena.workflow import Plot
from kolena.workflow.metrics import accuracy as compute_accuracy
from kolena.workflow.metrics import f1_score as compute_f1_score
from kolena.workflow.metrics import precision as compute_precision
from kolena.workflow.metrics import recall as compute_recall
from kolena.workflow.plot import Curve
from kolena.workflow.plot import CurvePlot


class BinaryClassificationEvaluator(BaseClassificationEvaluator):
    def compute_test_sample_metrics(
        self,
        ground_truth: GroundTruth,
        inference: Inference,
        threshold_configuration: ThresholdConfiguration,
    ) -> TestSampleMetricsSingleClass:
        if len(inference.inferences) != 1:
            log.warn(
                f"detected more than one ({len(inference.inferences)}) predicted labels for a binary "
                "classification test suite — aborting test",
            )
            raise ValueError

        threshold = threshold_configuration.threshold
        if threshold is None:
            threshold = 0.5

        prediction = inference.inferences[0]
        is_positive_prediction = prediction.score >= threshold
        is_positive_sample = ground_truth.classification.label == prediction.label
        return TestSampleMetricsSingleClass(
            classification=prediction,
            threshold=threshold,
            is_correct=is_positive_prediction == is_positive_sample,
            is_TP=is_positive_sample and is_positive_prediction,
            is_FP=not is_positive_sample and is_positive_prediction,
            is_FN=is_positive_sample and not is_positive_prediction,
            is_TN=not is_positive_sample and not is_positive_prediction,
        )

    def compute_test_case_metrics(
        self,
        ground_truths: List[GroundTruth],
        metrics_test_samples: List[TestSampleMetricsSingleClass],
    ) -> TestCaseMetricsSingleClass:
        n_tp = sum([mts.is_TP for mts in metrics_test_samples])
        n_fn = sum([mts.is_FN for mts in metrics_test_samples])
        n_fp = sum([mts.is_FP for mts in metrics_test_samples])
        n_tn = sum([mts.is_TN for mts in metrics_test_samples])

        return TestCaseMetricsSingleClass(
            TP=n_tp,
            FP=n_fp,
            FN=n_fn,
            TN=n_tn,
            Accuracy=compute_accuracy(n_tp, n_fp, n_fn, n_tn),
            Precision=compute_precision(n_tp, n_fp),
            Recall=compute_recall(n_tp, n_fn),
            F1=compute_f1_score(n_tp, n_fp, n_fn),
            FPR=n_fp / (n_fp + n_tn) if n_fp + n_tn > 0 else 0,
        )

    def compute_test_case_plots(
        self,
        ground_truths: List[GroundTruth],
        inferences: List[Inference],
        metrics: List[TestSampleMetricsSingleClass],
        gt_labels: List[str],
        confidence_range: Optional[Tuple[float, float, int]],
    ) -> List[Plot]:
        plots: List[Plot] = []

        if confidence_range:
            plots.extend(self._compute_test_case_confidence_histograms(metrics, confidence_range))
        else:
            log.warn("skipping test case confidence histograms: unsupported confidence range")

        positive_label = next(iter({inf.inferences[0].label for inf in inferences}))
        negative_label_set = set(gt_labels) - {positive_label}
        if len(negative_label_set) > 0:
            negative_label = next(iter(negative_label_set))
        else:
            # for a test suite with all positive samples, there is no way to obtain correct negative label.
            negative_label = "NOT " + positive_label

        plots.append(
            compute_roc_curves(
                [gt.classification for gt in ground_truths],
                [inf.inferences for inf in inferences],
                labels=[positive_label],
            ),
        )

        plots.append(
            compute_confusion_matrix(
                [gt.classification.label for gt in ground_truths],
                [positive_label if metric.is_TP or metric.is_FP else negative_label for metric in metrics],
                labels=[positive_label, negative_label],
            ),
        )

        threshold_curves = compute_threshold_curves(
            [gt.classification for gt in ground_truths],
            [inf.inferences[0] for inf in inferences],
        )
        if threshold_curves is not None and len(threshold_curves) == 3:
            plots.append(
                CurvePlot(
                    title="F1-Score vs. Confidence Threshold",
                    x_label="Confidence Threshold",
                    y_label="F1-Score",
                    curves=[threshold_curves[2]],
                ),
            )
            precisions = threshold_curves[0].y
            recalls = threshold_curves[1].y
            plots.append(
                CurvePlot(
                    title="Precision vs. Recall",
                    x_label="Recall",
                    y_label="Precision",
                    curves=[Curve(x=recalls, y=precisions, label=positive_label)],
                ),
            )

        plots = list(filter(lambda plot: plot is not None, plots))

        return plots
