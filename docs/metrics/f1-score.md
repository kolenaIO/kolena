# F<sub>1</sub> Score

The **F<sub>1</sub> score**, also known as **balanced F score** or **F measure**, is a metric that combines two
competing metrics, [precision](./precision.md) and [recall](./recall.md), with an equal weight. The F<sub>1</sub> score
symmetrically represents both precision and recall as one metric.

!!! info "Guide: Precision and Recall"

    Read the [precision](./precision.md) and the [recall](./recall.md) guide if you're not familiar with these
    terminologies.


Precision and recall offer a trade-off: if you increase precision, you reduce recall, and vice versa. This is called
the **precision/recall trade-off**. Ideally, we want to maximize both precision and recall to obtain the perfect model,
which is where the F<sub>1</sub> score comes in play. Because the F<sub>1</sub> score combines precision and recall
using [**harmonic mean**](https://en.wikipedia.org/wiki/Harmonic_mean), maximizing the F<sub>1</sub> score implies
simultaneously maximizing both precision and recall. Thus, the F<sub>1</sub> score has become a popular metric for the
evaluation of many workflows, such as classification, object detection, semantic segmentation, and information retrieval.

<div class="grid cards" markdown>
- :kolena-manual-16: API Reference: [`f1 score`][kolena.workflow.metrics.f1_score] ↗
</div>

## Implementation Details

Using [TP / FP / FN / TN](./tp-fp-fn-tn.md), we can define [precision](./precision.md) and [recall](./recall.md).
The F<sub>1</sub> score is computed simply by taking the **harmonic mean** of **precision** and **recall**.

The F<sub>1</sub> score is defined:

$$
\begin{align}
\text{F}_1 &= \frac {2} {\frac {1} {\text{precision}} + \frac {1} {\text{recall}}} \\[1em]
&= \frac {2 \times \text{precision} \times \text{recall}} {\text{precision} + \text{recall}}
\end{align}
$$

It can also be calculated directly from **true positive** (TP) / **false positive** (FP) / **false negative** (FN):

$$
\text{F}_1 = \frac {\text{TP}} {\text{TP} + \frac 1 2 \left( \text{FP} + \text{FN} \right)}
$$

### Examples

Perfect inferences:

<div class="grid" markdown>
| Metric | Value |
| --- | --- |
| TP | 20 |
| FP | 0 |
| FN | 0 |

$$
\begin{align}
\text{Precision} &= \frac{20}{20 + 0} \\[1em]
&= 1.0 \\[1em]
\text{Recall} &= \frac{20}{20 + 0} \\[1em]
&= 1.0 \\[1em]
\text{F}_1 &= \frac{20}{20 + \frac 1 2 \left( 0 + 0 \right)} \\[1em]
&= 1.0
\end{align}
$$
</div>


Partially correct inferences but every ground truth is recalled by an inference:

<div class="grid" markdown>
| Metric | Value |
| --- | --- |
| TP | 25 |
| FP | 75 |
| FN | 0 |

$$
\begin{align}
\text{Precision} &= \frac{25}{25 + 75} \\[1em]
&= 0.25 \\[1em]
\text{Recall} &= \frac{25}{25 + 0} \\[1em]
&= 1.0 \\[1em]
\text{F}_1 &= \frac{25}{25 + \frac 1 2 \left( 75 + 0 \right)} \\[1em]
&= 0.4
\end{align}
$$
</div>

Perfect inferences but some ground truths are missed:
<div class="grid" markdown>
| Metric | Value |
| --- | --- |
| TP | 25 |
| FP | 0 |
| FN | 75 |

$$
\begin{align}
\text{Precision} &= \frac{25}{25 + 0} \\[1em]
&= 1.0 \\[1em]
\text{Recall} &= \frac{25}{25 + 75} \\[1em]
&= 0.25 \\[1em]
\text{F}_1 &= \frac{25}{25 + \frac 1 2 \left( 0 + 75 \right)} \\[1em]
&= 0.4
\end{align}
$$
</div>

Zero correct inferences with non-zero false positive and false negative:
<div class="grid" markdown>
| Metric | Value |
| --- | --- |
| TP | 0 |
| FP | 10 |
| FN | 10 |

$$
\begin{align}
\text{Precision} &= \frac{0}{10 + 0} \\[1em]
&= 0.0 \\[1em]
\text{Recall} &= \frac{0}{10 + 0} \\[1em]
&= 0.0 \\[1em]
\text{F}_1 &= \frac{0}{0 + \frac 1 2 \left( 10 + 10\right)} \\[1em]
&= 0.0
\end{align}
$$
</div>

Zero correct inferences with zero false positive and false negative:
<div class="grid" markdown>
<div markdown>
| Metric | Value |
| --- | --- |
| TP | 0 |
| FP | 0 |
| FN | 0 |

!!! warning "Undefined F<sub>1</sub>"

    This example shows an edge case where both precision and recall are `undefined`. When either metric is `undefined`,
    F<sub>1</sub> is also `undefind`. In such cases, it's often interpreted as `0.0` instead.

</div>
$$
\begin{align}
\text{Precision} &= \frac{0}{0 + 0} \\[1em]
&= \text{undefined} \\[1em]
\text{Recall} &= \frac{0}{0 + 0} \\[1em]
&= \text{undefined} \\[1em]
\text{F}_1 &= \frac{0}{0 + \frac 1 2 \left( 0 + 0\right)} \\[1em]
&= \text{undefined} \\[1em]
\end{align}
$$
</div>


### Multiple Classes

In a **multi-class** case, the F<sub>1</sub> score can be computed per class. In the [TP / FP / FN / TN](./tp-fp-fn-tn.md)
guide, we went over multi-class cases and how these metrics are computed. Once you have these four metrics computed per
class, you can compute precision, recall, and F<sub>1</sub> score for each class by treating each as a single-class
problem.

### Aggregating Per-class Metrics

If you are looking for a **single** F<sub>1</sub> score that summarizes model performance across all classes, there are
different ways to aggregate per-class F<sub>1</sub> scores: **macro**, **micro**, and **weighted**. Read more about
these methods in the [Averaging Methods](./averaging-methods.md) guide.

## Limitations and Biases

While the F<sub>1</sub> score can be used to evaluate classification/object detection models with a single metric,
this metric is not adequate to use for all applications. In some applications, such as identifying pedestrians from an
autonomous vehicle, any false negatives can be life threatening. In these scenarios, having a few more false positives
as a trade-off for reducing the chance of any life-threatening events happening is preferred. Here, recall should be
weighted much more than the precision as it minimizes false negatives. To address the significance of recall,
**$\text{F}_\beta$ score** can be used as an alternative.


The **$\text{F}_\beta$ score** is a generic form of the F<sub>1</sub> score with a weight parameter, $\beta$, where
recall is considered $\beta$ times more important than precision:

$$
\text{F}_{\beta} = \frac {(1 + \beta^2) \times \text{precision} \times \text{recall}} {(\beta^2 \times \text{precision}) + \text{recall}}
$$

The three most common values for the beta parameter are as follows:

- **F<sub>0.5</sub> score** ($\beta$ = 0.5), where precision is more important than recall, it focuses more on minimizing FPs than minimizing FNs
- **F<sub>1</sub> score** ($\beta$ = 1), the true harmonic mean of precision and recall
- **F<sub>2</sub> score** ($\beta$ = 2), where recall is more important than precision, it focuses more on minimizing FNs than minimizing FPs
