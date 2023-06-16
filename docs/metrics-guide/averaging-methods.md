---
subtitle: Macro, Micro, Weighted
---

# Averaging Methods

For multiclass workflows like classification or object detection, metrics such as precision, recall, and F1-score are
computed **per class**. To compute a single value that represents model performance across all classes, these per-class
scores need to be aggregated. There are a few different averaging methods for doing this, most notably:

- [**macro**](#macro-average): unweighted mean of all per-class scores
- [**micro**](#micro-average): global average of per-sample TP, FP, FN scores
- [**weighted**](#weighted-average): mean of all per-class scores, weighted by sample sizes for each class

## Example: Multiclass Classification

Let’s consider the following multiclass classification metrics, computed across a total of 10 samples:

| Class | # Samples | <span title="# True Positives">TP</span> | <span title="# False Positives">FP</span> | <span title="# False Negatives">FN</span> | Precision | Recall | F1-score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Airplane | 3 | 2 | 1 | 1 | 0.67 | 0.67 | 0.67 |
| Boat | 1 | 1 | 3 | 0 | 0.25 | 1.0 | 0.4 |
| Car | 6 | 3 | 0 | 3 | 1.0 | 0.5 | 0.67 |

#### Macro Average

**Macro average** is perhaps the most straightforward among the numerous options and is computed by taking an
**unweighted** mean of all the per-class scores:

$$
\text{F1}_\text{macro} = \frac{0.67 + 0.4 + 0.67}{3} = 0.58
$$

#### Micro Average

In contrast to macro, **micro average** computes a **global** average by counting the sums of true positive (TP), false
negative (FN) and false positive (FP):

$$
\text{F1}_\text{micro} = \frac{6}{6 + 0.5 \times (4 + 4)} = 0.6
$$

But what about **micro precision** and **micro recall**?

<div class="grid" markdown>
$$
\text{Precision}_\text{micro} = \frac{6}{6 + 4} = 0.6
$$

$$
\text{Recall}_\text{micro} = \frac{6}{6 + 4} = 0.6
$$
</div>

Note that precision, recall, and f1-score all have the same value: $0.6$. This is because micro-averaging essentially
computes the proportion of correctly classified instances out of all instances, which is the definition of overall
**accuracy**.

In the multi-class classification cases where each sample has a single label, we get the following:

$$
\text{F1}_\text{micro} = \text{Precision}_\text{micro} = \text{Recall}_\text{micro} = \text{Accuracy}
$$

### Weighted Average

**Weighted average** computes the mean of all per-class scores while considering each class’s **support**. In this case,
support is the number of actual instances of the class in the dataset.

For example, if there are 5 samples of class `Airplane`, then the support value of class `Airplane` is 5. In other
words, support is the sum of true positive (TP) and false negative (FN) counts. The weight is the proportion of each
class’s support relative to the sum of all support values:

$$
\text{F1}_\text{weighted} = (0.67 \times 0.3) + (0.4 \times 0.1) + (0.67 \times 0.6) = 0.64
$$



## Intended Uses

You would generally use these three methods to aggregate the metrics computed per class. Averaging is most commonly used
in multiclass/multi-label classification and object detection tasks.

So which average should you use?

If you’re looking for an easily understandable metric for overall model performance regardless of class, **accuracy** or
**micro average** are probably best.

If you want to treat all classes equally, then using **macro average** would be a good choice.

If you have an imbalanced dataset but want to assign more weight to classes with more samples, consider using
**weighted average** instead of the **macro average** method.
