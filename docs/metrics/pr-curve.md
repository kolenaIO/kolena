---
search:
  exclude: true
---

# Precision-Recall (PR) Curve

!!! info inline end "Guides: Precision and Recall"

    Read the [precision](./precision.md) and the [recall](./recall.md) guides if you're not familiar with those metrics.

A precision-recall (PR) curve is a plot that gauges machine learning model performance by using
[precision](./precision.md) and [recall](./recall.md), which are performance metrics that evaluate the quality of a
classification model. The curve is built with precision on the y-axis and recall on the x-axis computed across many
thresholds, showing a trade-off of how precision and recall values change when a classification threshold changes.

<div class="grid cards" markdown>
- :kolena-manual-16: API Reference: [`CurvePlot` ↗][kolena.workflow.plot.CurvePlot]
</div>

## Implementation Details

The curve’s points (precisions and recalls) are calculated with a varying threshold, and made into points (precision
values on the y-axis and recall values on the x-axis). Precision and recall are threshold-dependent metrics where a
threshold value must be defined to compute them, and by computing and plotting these two metrics across many thresholds
we can check how these metrics change depending on the threshold.

??? info "Thresholds Selection"

    Threshold ranges are very customizable. Typically, a uniformly spaced range of values from 0 to 1 can model a PR
    curve, where users pick the number of thresholds to include. Another common approach to picking thresholds is
    collecting and sorting the unique confidences of every prediction.

### Example: Binary Classification

Let's consider a simple binary classification example and plot a PR curve at a uniformly spaced range of thresholds.
The table below shows six samples (four positive and two negative) sorted by their confidence score. Each inference
is evaluated at each threshold: 0.25, 0.5, and 0.75. It's a negative prediction if its confidence score is below the
evaluating threshold; otherwise, it's positive.

<center>

| Sample | <nobr>Confidence ↓</nobr> | Inference @ 0.25 | Inference @ 0.5 | Inference @ 0.75 |
| --- | --- | --- | --- | --- |
| Positive | 0.9 | <span style="color: green">Positive</span> | <span style="color: green">Positive</span> | <span style="color: green">Positive</span> |
| Positive | 0.8 | <span style="color: green">Positive</span> | <span style="color: green">Positive</span> | <span style="color: green">Positive</span> |
| Positive | 0.7 | <span style="color: green">Positive</span> | <span style="color: green">Positive</span> | <span style="color: red">Negative</span> |
| Negative | 0.4 | <span style="color: green">Positive</span> | <span style="color: green">Positive</span> | <span style="color: red">Negative</span> |
| Positive | 0.35 | <span style="color: green">Positive</span> | <span style="color: red">Negative</span> | <span style="color: red">Negative</span> |
| Negative | 0.3 | <span style="color: green">Positive</span> | <span style="color: red">Negative</span> | <span style="color: red">Negative</span> |

</center>

As the threshold increases, there are fewer false positives and more false negatives, most likely yielding high
precision and low recall. Conversely, decreasing the threshold may improve recall at the cost of precision. Let's
compute the precision and recall values at each threhold.

<center>

| Threshold | TP | FP | FN | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| **0.25** | 4 | 2 | 0 | $\frac{4}{6}$ | $\frac{4}{4}$ |
| **0.5** | 3 | 1 | 1 | $\frac{3}{4}$ | $\frac{3}{4}$ |
| **0.75** | 2 | 0 | 2 | $\frac{2}{2}$ | $\frac{2}{4}$ |

</center>

Using these precision and recall values, a PR curve can be plotted:

<center>
![pr.png](../assets/images/metrics-prcurve-example.png)
</center>

### Multiclass

For multiple classes, micro or macro precision and recall can be used instead. It is also useful to plot a curve **per
class** by binarizing the input so that we only consider the class of interest. This technique is known as
**one-vs-rest** (OvR). With this strategy, we can have `n` PR curves for `n` unique classes.

## Area Under the PR Curve (AUPRC)

The area under the PR curve (AUPRC) is a **threshold-independent** metric that summarizes the performance of a model
depicted by a PR curve. The greater the area, the better a model performs. The
[average precision](./average-precision.md) is one particular method for calculating the AUPRC. With PR curves, we
don’t have to go into any math and can visually conclude which curves indicate that a certain class or model has a
better performance.

<center>
![pr.png](../assets/images/metrics-prcurve.png)
</center>

In the plot above, we see that the blue curve has a higher precision than the orange curve for almost every recall
value. This means that the model behind the blue curve performs better. There are many factors that might make the
blue curve more jagged compared to the orange curve, such as having a lower number of thresholds (fewer points), poorly
chosen thresholds, insufficient amounts of data, or changed model performance for different threshold ranges.


## Limitations and Biases

PR curves are a very common plot used in practice to evaluate model performance in terms of precision and recall. There
are some pitfalls that might be overlooked: class imbalance, source of error, and poor threshold choices.

1. Classes with too few data points may have PR curves that are poor representations of actual performance or overall
performance. The performance of minority classes may be less accurate compared to a majority class.
2. PR curves only gauge precision and recall based on classifications, they do not surface misclassification patterns
or reasons for different types of errors.
3. The values of the thresholds affect the shape of PR curves, which can affect how they are interpreted. Having a
different number of thresholds, or having different threshold values, make PR curve comparisons difficult.
