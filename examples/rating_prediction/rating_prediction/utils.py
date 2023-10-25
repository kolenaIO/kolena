import numpy as np
from typing import List
from rating_prediction.workflow import GroundTruth
from rating_prediction.workflow import Inference

from kolena.workflow import Curve
from kolena.workflow.metrics import f1_score
from kolena.workflow.metrics import precision
from kolena.workflow.metrics import recall


def compute_threshold_curves(
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
) -> List[Curve]:
    thresholds = list(np.linspace(0, 1, 10))

    precisions = []
    recalls = []
    f1s = []
    for threshold in thresholds:
        gts = [gts.rating >= threshold for gts in ground_truths]
        infs = [inf.rating >= threshold for inf in inferences]
        tp = len([True for gt, inf in zip(gts, infs) if gt and inf])
        fp = len([True for gt, inf in zip(gts, infs) if not gt and inf])
        fn = len([True for gt, inf in zip(gts, infs) if gt and not inf])

        precisions.append(precision(tp, fp))
        recalls.append(recall(tp, fn))
        f1s.append(f1_score(tp, fp, fn))

    return [
        Curve(x=thresholds, y=precisions, label="Precision"),
        Curve(x=thresholds, y=recalls, label="Recall"),
        Curve(x=thresholds, y=f1s, label="F1"),
    ]
