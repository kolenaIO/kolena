---
search:
  exclude: true
---

# Accuracy

Accuracy is one of the most well-known metrics in machine learning model evaluation because it is simple to understand
and straightforward to calculate.

Accuracy measures how often a model correctly predicts something (ranging from 0 to 1, with 1 being perfect
inferences). It reports the ratio of the number of correct inferences to the total number of inferences,
making it a good metric for assessing model performance in simple cases with balanced data. However, accuracy is much
less meaningful with unbalanced datasets (e.g. far more negative samples than positive samples) and should be used with
caution.

## Implementation Details

Accuracy is generally used to evaluate classification models. Aside from classification, accuracy is also often
used to evaluate semantic segmentation models by measuring the percent of correctly classified pixels in an image.

In a classification task, accuracy is the ratio of the number of correct predictions to the total number of predictions.

With [TP / FP / FN / TN counts](./tp-fp-fn-tn.md) computed, accuracy is defined:

$$
\text{Accuracy} =  \frac {\text{TP} + \text{TN}} {\text{TP} + \text{FP} + \text{FN} + \text{TN}}
$$

### Examples

Perfect inferences across 20 samples:

<div class="grid" markdown>
| Metric | Value |
| --- | --- |
| TP | 10 |
| FP | 0 |
| FN | 0 |
| TN | 10 |

$$
\begin{align}
\text{Accuracy} &= \frac{10 + 10}{10 + 0 + 0 + 10} \\[1em]
&= 1.0
\end{align}
$$
</div>

Partially correct inferences across 20 samples:

<div class="grid" markdown>
| Metric | Value |
| --- | --- |
| TP | 8 |
| FP | 4 |
| FN | 2 |
| TN | 6 |

$$
\begin{align}
\text{Accuracy} &= \frac{8 + 6}{8 + 4 + 2 + 6} \\[1em]
&= 0.7
\end{align}
$$
</div>

Highly imbalanced data, with 99 negative samples and 10 positive samples, with _no_ positive inferences:

<div class="grid" markdown>
| Metric | Value |
| --- | --- |
| TP | 0 |
| FP | 0 |
| FN | 10 |
| TN | 990 |

<div markdown>
$$
\begin{align}
\text{Accuracy} &= \frac{0 + 990}{0 + 0 + 10 + 990} \\[1em]
&= 0.99
\end{align}
$$

!!! warning "Be careful with imbalanced datasets!"

    This example describes a trivial model that only ever returns negative inferences, yet it has the high accuracy
    score of 99%.
</div>
</div>

## Limitations and Biases

While accuracy generally describes a classifier’s performance, it is important to note that the metric can be deceptive,
especially when [the data is imbalanced](https://stephenallwright.com/imbalanced-data/).

For example, let’s say there
are a total of 500 samples, with 450 belonging to the positive class and 50 to the negative. If the model correctly
predicts all the positive samples but misses all the negative ones, its accuracy is `450 / 500 = 0.9`. An accuracy score
of 90% indicates a pretty good model — but is a model that fails 100% of the time on negative samples useful? Using the
accuracy metric alone can hide a model’s true performance, so we recommend other metrics that are better suited for
imbalanced data, such as:

- [Balanced accuracy](https://stephenallwright.com/balanced-accuracy/)
- Precision
- Recall
- F1 score
