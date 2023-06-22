---
search:
  exclude: true
---

# Precision

The precision metric is designed to measure the percentage of **positive predictions** that a model correctly predicted, ranging from 0 to 1 (where 1 is best).

![precision and recall from Wikipedia](../assets/images/metrics-precision-recall.png)
<p style="text-align: center; color: gray;">
    A diagram of precision and recall from
	<a href="https://en.wikipedia.org/wiki/Precision_and_recall#Precision">Wikipedia</a>
</p>

As shown in the image above, precision is the fraction of correct predictions among all positive ones: `TP / (TP + FP),` where `TP` is the number of true positives and `FP` is the number of false positives.

In other words, precision measures the ability of the classifier to not label a negative sample as positive.

Read this [guide](./tp-fp-fn-tn.md) if you are not familiar with TP, FP, FN and TN.


## Implementation Details
The precision metric is generally used to evaluate classification models, especially when the objective is to reduce the **false positives** in predictions.

Aside from classification, it has also been commonly used in object detection, semantic segmentation, and information retrieval.

In a binary classification task, the precision metric is simply a ratio of the number of correct positive predictions to the total number of positive predictions.

 Consider the following notation, where

- $y_{true}$ is the list of ground truth
- $y_{pred}$ is the list of prediction

Then the metric is defined as

$$
precision(y_{true}, y_{pred}) = \frac {TP} {TP + FP}
$$

**Example of perfect precision**

```python
>>> y_true = [0, 0, 1, 1, 1, 1]
>>> y_pred = [0, 0, 0, 0, 1, 1]
>>> print(f"precision: {precision(y_true, y_pred)}")
precision: 1.0
```

**Example of some incorrect predictions**

```python
>>> y_true = [0, 0, 0, 1, 1, 1]
>>> y_pred = [0, 1, 1, 0, 1, 1]
>>> print(f"precision: {precision(y_true, y_pred)}")
precision: 0.5
```

**Example of zero positive predictions**

```python
>>> y_true = [0, 0, 0, 0, 1, 1, 1, 1]
>>> y_pred = [0, 0, 0, 0, 0, 0, 0, 0]
>>> print(f"precision: {precision(y_true, y_pred)}")
precision: 0.0
```

### Multiple Classes

So far, we have only looked at **binary** classification/object detection cases, but in **multi-class** or **multi-label** cases, precision is computed per class. In the [**TP / FP / FN / TN**](./tp-fp-fn-tn.md) metrics guide, we went over multiple-class cases and how these metrics are computed. Once you have these four metrics computed per class, you can compute precision for each class by treating it as a binary class problem.

### Aggregating Per-class Metrics

If you are looking for a **single** precision score that summarizes model performance across all classes, there are different ways to aggregate per-class precision scores: **macro**, **micro**, **weighted,** and **samples**. You can read more on different averaging methods in [this guide](./averaging-methods.md).

# Limitations and Biases

As shown in the formula above, precision only uses TP and FP; TN and FN are not taken into account. Thus, precision should only be used in situations where correct identification of the negative class and incorrect identification of the positive class do not play a role. This is why this metric is mainly used for object detection and information retrieval: **failing to detect negative samples has no consequences**.

Precision is a particularly bad measure when there are only a few predictions that belong to the positive class because it will result in high precision even when the model fails to predict most of the positive samples. In such scenarios, you should use the [recall](./recall.md) metric instead.

## Kolena API

[`kolena.workflow.metrics.recall`](https://docs.kolena.io/reference/workflow/metrics/#kolena.workflow.metrics.recall)
