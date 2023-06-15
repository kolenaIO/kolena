# Averaging Methods

## Description

In the case of multi-class classification/object detection, you have to compute metrics such as precision, recall, and f1-score **per class**. When you want a single value that represents model performance across all classes, these per-class scores need to be aggregated. There are different averaging methods for doing this, namely, **macro**, **micro,** **weighted,** and **samples**.

Let’s consider the following multi-class classification metrics:

| Classes | TP | FP | FN | Precision | Recall | F1-score |
| --- | --- | --- | --- | --- | --- | --- |
| Airplane | 2 | 1 | 1 | 0.67 | 0.67 | 0.67 |
| Boat | 1 | 3 | 0 | 0.25 | 1.0 | 0.4 |
| Car | 3 | 0 | 3 | 1.0 | 0.5 | 0.67 |

**Macro-averaging** is perhaps the most straightforward among the numerous options and is computed by taking an **unweighted** mean of all the per-class scores:

```python
macro-avg F1 = (0.67 + 0.4 + 0.67) / 3 = 0.58
```

In contrast, **micro-averaging** computes a **global** average by counting the sums of true positive (TP), false negative (FN) and false positive (FP):

```python
micro-avg F1 = 6 / (6 + 0.5 * (4 + 4)) = 0.6
```

But what about `micro-avg precision` and `micro-avg recall`?

```python
micro-avg precision = 6 / (6 + 4) = 0.6
micro-avg recall = 6 / (6 + 4) = 0.6
```

Note that precision, recall, and f1-score all have the same value (0.6). This is because micro-averaging essentially computes the proportion of correctly classified instances out of all instances, which is the definition of overall **accuracy**. 

In the multi-class classification cases where each sample has a single label, we get the following:

```python
 micro-F1 = micro-precision = micro-recall = accuracy
```

**Weighted averaging** computes the mean of all per-class scores while considering each class’s **support**. In this case, support is the number of actual instances of the class in the dataset, for example, if there are 5 samples of class **Airplane**, then the support value of class **Airplane** is 5. In other words, support is the sum of TP and FN counts. The weight is the proportion of each class’s support relative to the sum of all support values:

```python
weighted-avg F1 = (0.67  * 0.3) + (0.4 * 0.1) + (0.67 * 0.6) = 0.64
```

**Samples averaging** computes the mean of metrics computed per sample. It’s only applicable to multi-label classification cases where a sample has multiple labels. 

## Intended Uses

You would generally use these four methods to aggregate the metrics computed per class. Averaging is most commonly used in multi-class/multi-label classification or object detection tasks.

So which average should you use? 

If you’re looking for an easily understandable metric for overall model performance regardless of class, **accuracy** or **micro-average** are probably best.

If you want to treat all classes equally, then using **macro-average** would be a good choice. 

If you have an imbalanced dataset but want to assign more weight to classes with more samples, consider using **weighted average** instead of the **macro-average** method.