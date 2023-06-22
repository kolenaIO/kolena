---
icon: kolena/metrics-glossary-16
# TODO: remove search exclusion before landing Metrics Glossary
search:
  exclude: true
---

# :kolena-metrics-glossary-20: Metrics Glossary

This section contains guides for different metrics used to measure model performance.

Each ML use case requires different metrics. Using the right metrics is critical for understanding and meaningfully
comparing model performance. In each metrics guide, you can learn about the metric with examples, its limitations and
biases, and its intended uses.

<div class="grid cards" markdown>
- [Averaging Methods: Macro, Micro, Weighted](averaging-methods.md)

    ---

    Different averaging methods for aggregating metrics for **multiclass** workflows, such as classification and
    object detection.
</div>

<div class="grid cards" markdown>
- [Geometry Matching](geometry-matching.md)

    ---

    Geometry matching is the process of matching inferences to ground truths for computer vision workflows with a
    localization component. It is a core building block for metrics such as TP, FP, and FN, and any metrics built on
    top of these, like precision, recall, and F1 score.
</div>

<div class="grid cards" markdown>
- [Intersection over Union (IoU)](iou.md)

    ---

    This metric measures overlap between two geometries, segmentation masks, sets of labels, or time-series snippets.
    Also known as Jaccard index in classification workflow.
</div>
