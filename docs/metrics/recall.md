---
search:
  exclude: true
---

# Recall / TPR

The recall metric, also known as true positive rate (TPR) and sensitivity, is designed to measure the percentage of **positive instances** that a model correctly predicted, ranging from 0 to 1 (where 1 is best).

![precision and recall from Wikipedia](../assets/images/metrics-precision-recall.png)
<p style="text-align: center; color: gray;">
    A diagram of precision and recall from
	<a href="https://en.wikipedia.org/wiki/Precision_and_recall#Precision">Wikipedia</a>
</p>

As shown in the image above, recall is the fraction of correct predictions along all positive samples: `TP / (TP + FN)`, where `TP` is the number of true positives and `FN` is the number of false negatives.

In other words, the recall metric measures a classifier’s ability to find all positive samples.

Read this [guide](./tp-fp-fn-tn.md) if you are not familiar with TP, FP, FN and TN.


## Implementation Details
The recall metric is generally used to evaluate classification models, especially when the objective is to reduce the **false** **negatives** in positive samples**.**

Aside from classification, it is also commonly used in object detection, semantic segmentation, and information retrieval.

In a binary classification task, the recall metric is simply a ratio of the number of correct positive predictions to the total number of positive samples.

Consider the following notation, where

- $y_{true}$ is the list of ground truth
- $y_{pred}$ is the list of prediction

Then the metric is defined as

$$
recall(y_{true}, y_{pred}) = \frac {TP} {TP + FN}
$$

**Example of perfect recall**

```python
>>> y_true = [0, 0, 0, 1, 1, 1]
>>> y_pred = [0, 1, 1, 1, 1, 1]
>>> print(f"recall: {recall(y_true, y_pred)}")
recall: 1.0
```

**Example of some incorrectly predicted positive samples**

```python
>>> y_true = [0, 0, 1, 1, 1, 1]
>>> y_pred = [0, 0, 0, 0, 1, 1]
>>> print(f"recall: {recall(y_true, y_pred)}")
recall: 0.5
```

**Example of zero positive samples**

```python
>>> y_true = [0, 0, 0, 0, 0, 0]
>>> y_pred = [0, 0, 0, 0, 1, 1]
>>> print(f"recall: {recall(y_true, y_pred)}")
recall: 0.0
```

### Multiple Classes

So far, we’ve only looked at binary classification/object detection cases, but in the **multi-class** or **multi-label** cases, recall is computed per class. In the [TP / FP / FN / TN](./tp-fp-fn-tn.md) metrics guide, we went over multiple-class cases and how these metrics are computed. Once you have them computed per class, you can compute recall for each class by treating it as a binary class problem.

### Aggregating Per-class Metrics

If you are looking for a **single** precision score that summarizes model performance across all classes, there are different ways to aggregate these per-class precision scores: **macro**, **micro**, **weighted,** and **samples**. You can read more on different averaging methods in [this guide](./averaging-methods.md).

## Limitations and Biases

As shown in the formula above, recall only uses TP and FN; TN and FP are not taken into account. Thus, recall should only be used in situations where performance on the negative class does not play a role. This is why this metric is mainly used for object detection and information retrieval: **failing to detect negative samples has no consequences**.

Recall is a particularly bad measure when there are only a few positive samples because it will result in high recall when the model fails to predict most of the negative samples. In such scenarios, you should use the [precision](./precision.md) metric instead.

## Kolena API

[`kolena.workflow.metrics.recall`](https://docs.kolena.io/reference/workflow/metrics/#kolena.workflow.metrics.recall)
