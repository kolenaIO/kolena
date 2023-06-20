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
- [Geometry Matcher](geometry-matcher.md)

    ---

    This algorithm finds the best possible match, given the sets of the ground truth and prediction polygons for each image.
    It is a building block for any object detection metrics.
</div>

<div class="grid cards" markdown>
- [Intersection over Union (IoU)](iou.md)

    ---

    This metric measures overlap between two geometries, segmentation masks, sets of labels, or time-series snippets.
    Also known as Jaccard index in classification workflow.
</div>
