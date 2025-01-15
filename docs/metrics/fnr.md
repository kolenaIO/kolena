---
description: Understanding false negative rate (FNR) to evaluate ML model performance
---


# False Negative Rate (FNR)

It  by the model.
False negative rate (FNR) measures the proportion of actual positive instances
that are incorrectly classified as negative, ranging from 0 to 1. A low false negative rate indicates that the model is
good at identifying positive cases, where a high false positive rate suggests that the model is
frequently missing positive cases, which can be critical in applications like medical diagnoses.

As shown in this diagram, false positive rate is the fraction of all negative ground truths that are incorrectly predicted:

$$\text{FNR} = \frac{\text{FN}}{\text{FN} + \text{TP}}$$

In the above formula, $\text{FN}$ is the number of false negative inferences and $\text{TP}$ is the number of true
positive inferences.

!!! info "Guide: True Negative / False Positive"

    Read the [TP / FP / FN / TN](./tp-fp-fn-tn.md) guide if you're not familiar with "FN" and "TP" terminology.

## Implementation Details

FNR is used to evaluate the performance of a classification model, particularly in tasks like binary
classification, where the goal is to classify data into one of two possible classes.

Here is how FPR is calculated:

$$
\text{FPR} = \frac {\text{# False Negatives}} {\text{# False Negatives} + \text{# True Positives}}
$$

### Multiple Classes

So far, we have only looked at **binary** classification cases, but in **multiclass** or **multi-label** cases,
FNR is computed per class. In the [TP / FP / FN / TN](./tp-fp-fn-tn.md) guide,
we went over multiple-class cases and how these metrics are computed. Once you have these four metrics computed per
class, you can compute FNR for each class by treating each as a single-class problem.

### Aggregating Per-class Metrics

If you are looking for a **single** FNR score that summarizes model performance across all classes, there are
different ways to aggregate per-class FNR scores: **macro**, **micro**, and **weighted**. Read more about these
methods in the [Averaging Methods](./averaging-methods.md) guide.
