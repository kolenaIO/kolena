---
search:
  exclude: true
---

# TP / FP / FN / TN

The counts of **true positive** (TP), **false positive** (FP), **false negative** (FN), and **true negative** (TN)
predictions and ground truths are essential for summarizing model performance. These metrics are the building blocks of
many other metrics, including accuracy, precision, and recall.

| Metric | | Description |
| --- | --- | --- |
| True Positive | TP | An instance for which both predicted and actual values are positive |
| False Positive | FP | An instance for which predicted value is positive but actual value is negative |
| <nobr>False Negative</nobr> | FN | An instance for which predicted value is negative but actual value is positive |
| True Negative | TN | An instance for which both predicted and actual values are negative |

To compute these metrics, each prediction is compared to a ground truth and categorized into one of the four groups.
Let’s say we’re building a dog classifier that predicts whether an image has a dog or not:

![A dog classifier example](../assets/images/metrics-tpfpfntn-dog-classifier.png)

Images of a dog are **positive** samples, and images without a dog are **negative** samples.

If a classifier predicts that there is a dog on a **positive** sample, that prediction is a **true positive** (TP).
If that classifier predicts that there isn’t a dog on a positive sample, that prediction is a **false negative** (FN).

Similarly, if that classifier predicts that there is a dog on a **negative** sample, that prediction is a
**false positive** (FP). A negative prediction on a negative sample is a **true negative** (TN).

## Implementation Details

The TP / FP / FN / TN metrics have been around for a long time and are mainly used to evaluate classification,
detection, and segmentation models.

The implementation of these metrics is simple and straightforward. That said, there are different guidelines and edge
cases to be aware of for binary and multiclass problems as well as object detection and other workflows.

### Classification

There are three types of classification tasks: **binary**, **multi-class,** and **multi-label**.

#### Binary

In a binary classification task, TP, FN, FP, and TN are implemented as follows:

| Variable | Type | Description |
| --- | --- | --- |
| <nobr>`ground_truths`</nobr> | `List[bool]` | Ground truth labels, where `True` indicates a positive sample |
| `inferences` | <nobr>`List[float]`</nobr> | Predicted confidence scores, where a higher score indicates a higher confidence of the sample being positive |
| `T` | `float` | Threshold value to compare against the prediction’s confidence score, where `score >= T` is positive |

With these inputs, TP / FP/ FN / TN metrics are defined:

```python
TP = sum(    gt and inf >= T for gt, inf in zip(ground_truths, inferences))
FP = sum(not gt and inf >= T for gt, inf in zip(ground_truths, inferences))
FN = sum(    gt and inf <  T for gt, inf in zip(ground_truths, inferences))
TN = sum(not gt and inf <  T for gt, inf in zip(ground_truths, inferences))
```

??? example "Example: Binary Classification"

    This example considers five samples with the following ground truths, inferences, and threshold:

    ```python
    ground_truths = [False, True, False, False, True]
    inferences = [0.3, 0.2, 0.9, 0.4, 0.5]
    T = 0.5
    ```

    Using the above formula for TP, FP, FN, and TN yields the following metrics:

    ```python
    print(f"TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    # TP=1, FN=1, FP=1, TN=2
    ```


#### Multiclass

TP / FP / FN / TN metrics are computed a little differently in **multiclass** classification tasks.

For multiclass classification tasks, these four metrics are defined **per class**. This technique,
also known as **one-vs-rest** (OvR), essentially evaluates each class as a binary classification problem.

Consider a classification problem where a given image belongs to either the `Airplane`, `Boat`, or `Car` class. Each of
these TP / FP / FN / TN metrics is computed for each class. For class `Airplane`, the metrics are defined as follows:

| Metric | Example |
| --- | --- |
| True Positive | Any image predicted as an `Airplane` that is labeled as an `Airplane` |
| False Positive | Any image predicted as an `Airplane` that is _not_ labeled as an `Airplane` (e.g. labeled as `Boat` but predicted as `Airplane`) |
| <nobr>False Negative</nobr> | Any image _not_ predicted as an `Airplane` that is labeled as an `Airplane` (e.g. labeled as `Airplane` but predicted as `Car`) |
| True Negative | Any image _not_ predicted as an `Airplane` that is _not_ labeled as an `Airplane` (e.g. labeled as `Boat` but predicted as `Boat` or `Car`) |

#### Multi-label

In a **multi-label** classification task, TP / FP / FN / TN are computed per class, like in multiclass classification.

A sample is considered to be a positive one if the ground truth **includes** the evaluating class; otherwise, it’s a
negative sample. The same logic can be applied to the predictions, so, for example, if a classifier predicts that this
sample belongs to class `Airplane` and `Boat`, and the ground truth for the same sample is only class `Airplane`, then
this sample is considered to be a TP for class `Airplane`, and FP for class `Boat`.

Multi-label classification tasks can alternately be thought of as a collection of binary classification tasks.

### Object Detection

There are some differences in how these four metrics work for a detection task compared to a classification task.
Rather than being computed at the sample level (e.g. per image), they're computed at the instance level (i.e. per object)
for instances that the model is detecting. When given an image with multiple objects, each inference and each ground truth
is assigned to one group, and the definitions of the terms are slightly altered:

| Metric | | Description |
| --- | --- | --- |
| True Positive | TP | Inference that is matched with a ground truth and has a confidence score $\geq$ threshold |
| False Positive | FP | Inference that is not matched with a ground truth and has a confidence score $\geq$ threshold |
| <nobr>False Negative</nobr> | FN | Ground truth that is not matched with an inference or that is matched with an inference that has a confidence score $<$ threshold |
| True Negative | TN | <p>:kolena-warning-sign-16: **Poorly defined for object detection!** :kolena-warning-sign-16:</p><p>In object detection tasks, a true negative is any non-object that isn't detected as an object. This isn't well defined and as such true negative isn't a commonly used metric in object detection.</p><div>Occasionally, for object detection tasks "true negative" is used to refer to any image that does not have any true positive or false positive inferences.</div> |

In an object detection task, checking for detection correctness requires a couple of other metrics (e.g., [Intersection
over Union (IoU)](./iou.md) and [Geometry Matching](./geometry-matching.md)).

#### Single-class

Let’s assume that a [matching algorithm](./geometry-matching.md) has already been run on all predictions and that the matched pairs and unmatched
ground truths and predictions are given. Consider the following variables, adapted from
[`match_inferences`][kolena.workflow.metrics.match_inferences]:

| Variable | Type | Description |
| --- | --- | --- |
| `matched` | <nobr>`List[Tuple[GT, Inf]]`</nobr> | List of **matched** ground truth and prediction bounding box **pairs** |
| `unmatched_gt` | `List[GT]` | List of **unmatched** ground truth bounding boxes |
| <nobr>`unmatched_inf`</nobr> | `List[Inf]` | List of **unmatched** inference bounding boxes |
| `T` | `float` | Threshold used to filter valid prediction bounding boxes based on their confidence scores |

Then these metrics are defined:

```python
TP = len([inf.score >= T for _, inf in matched])
FN = len([inf.score <  T for _, inf in matched]) + len(unmatched_gt)
FP = len([inf.score >= T for inf in unmatched_inf])
```

??? example "Example: Single-class Object Detection"

    ![Legends](../assets/images/metrics-tpfpfntn-legends.png)

    ![Single-class example](../assets/images/metrics-tpfpfntn-single-class.png)

    This example includes two ground truths and two inferences, and when computed with an IoU threshold of 0.5 and
    confidence score threshold of 0.5 yields:

    | TP | FP | FN |
    | --- | --- | --- |
    | 1 | 1 | 1 |

#### Multiclass

Like classification, multiclass object detection tasks compute TP / FP / FN per class.

??? example "Example: Multiclass Object Detection"

    ![Legends](../assets/images/metrics-tpfpfntn-legends.png)

    ![Multi-class example](../assets/images/metrics-tpfpfntn-multi-class.png)

    Similar to multiclass classification, TP / FP / FN are computed for class `Apple` and class `Banana` separately.

    Using an IoU threshold of 0.5 and a confidence score threshold of 0.5, this example yields:

    | Class | TP | FP | FN |
    | --- | --- | --- | --- |
    | `Apple` | 0 | 0 | 1 |
    | `Banana` | 0 | 1 | 0 |

### Averaging Per-class Metrics

For problems with multiple classes, these TP / FP / FN / TN metrics are computed for each class. If you are looking for
a single score that summarizes model performance across all classes, there are a few different ways to aggregate
per-class metrics: **macro**, **micro**, and **weighted.**

Read more about these different averaging methods in the [Averaging Methods guide](./averaging-methods.md).

## Limitations and Biases

TP, FP, TN, and FN are four metrics based on the assumption that each sample/instance can be classified as a positive
or a negative, thus they can only be applied to single-class applications. The workaround for multiple-class
applications is to compute these metrics for each label using the **one-vs-rest** (OvR) strategy and then treat it
as a single-class problem.

Additionally, these four metrics don't take model confidence score into account. All inferences above the confidence
score threshold are treated the same! For example, when using a confidence score threshold of 0.5, a prediction with a
confidence score barely above the threshold (e.g. 0.55) is treated the same as a prediction with a very high confidence
score (e.g. 0.99). In other words, any inference above the confidence threshold is considered as a positive prediction.
To examine performance taking confidence score into account, consider plotting a histogram of the distribution of
confidence scores.
