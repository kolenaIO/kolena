---
icon: kolena/metrics-glossary-16
hide:
  - toc
---

# :kolena-metrics-glossary-20: Metrics Glossary

This section contains guides for different metrics used to measure model performance.

Each ML use case requires different metrics. Using the right metrics is critical for understanding and meaningfully
comparing model performance. In each metrics guide, you can learn about the metric with examples, its limitations and
biases, and its intended uses.

<div class="grid cards" markdown>
- [Accuracy](accuracy.md)

    ---

    Accuracy measures how well a model predicts correctly. It's a good metric for assessing model performance in simple
    cases with balanced data.

- [Averaging Methods: Macro, Micro, Weighted](averaging-methods.md)

    ---

    Different averaging methods for aggregating metrics for **multiclass** workflows, such as classification and
    object detection.

- [F<sub>1</sub> score](f1-score.md)

    ---

    F<sub>1</sub> score is a metric that combines two competing metrics, [precision](precision.md) and
    [recall](recall.md) with an equal weight. It symmetrically represents both precision and recall as one metric.

- [Geometry Matching](geometry-matching.md)

    ---

    Geometry matching is the process of matching inferences to ground truths for computer vision workflows with a
    localization component. It is a core building block for metrics such as [TP, FP, and FN](tp-fp-fn-tn.md), and any
    metrics built on top of these, like [precision](precision.md), [recall](recall.md), and
    [F<sub>1</sub> score](f1-score.md).

- [Intersection over Union (IoU)](iou.md)

    ---

    IoU measures overlap between two geometries, segmentation masks, sets of labels, or time-series snippets.
    Also known as Jaccard index in classification workflow.

- [Precision](precision.md)

    ---

    Precision measures the proportion of positive inferences from a model that are correct. It is useful when the
    objective is to measure and reduce false positive inferences.

- [Recall](recall.md)

    ---

    Recall, also known as true positive rate (TPR) and sensitivity, measures the proportion of all positive ground
    truths that a model correctly predicts. It is useful when the objective is to measure and reduce false negative
    ground truths, i.e. model misses.

- [TP / FP / FN / TN](tp-fp-fn-tn.md)

    ---

    The counts of TP, FP, FN and TN ground truths and inferences are essential for summarizing model performance. They
    are the building blocks of many other metrics, including [accuracy](accuracy.md), [precision](precision.md),
    and [recall](recall.md).
</div>
