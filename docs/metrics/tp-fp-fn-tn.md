# TP / FP / FN / TN

## Description

The counts of **true positive** (TP), **false positive** (FP), **false negative** (FN), and **true negative** (TN) are essential metrics for summarizing predictions and ground truths, thereby letting you measure a classification task’s performance. These metrics are the building blocks of many other metrics, including accuracy, precision, and recall.

| TP | An instance for which both predicted and actual values are positive |
| --- | --- |
| FP | An instance for which predicted value is positive but actual value is negative |
| FN | An instance for which predicted value is negative but actual value is positive |
| TN | An instance for which both predicted and actual values are negative |

Each prediction is compared to a ground truth to be categorized into one of the four groups. Let’s say we’re building a dog classifier that predicts whether an image has a dog or not:

![A dog classifier example](../assets/images/metrics-tpfpfntn-dog-classifier.png)

Images of a dog are **positive** samples, and images without a dog are **negative** samples, so if the classifier predicts that there is a dog on a **positive** sample, then it’s a **TP**.  If it predicts that there isn’t a dog on a positive sample, then it’s an **FN**.

Similarly, if it predicts that there is a dog on a **negative** sample, then it’s an **FP**.  A negative prediction on a negative sample is a **TN**.

## Intended Uses

These terminologies have been around for a long time and are mainly used to evaluate classification and detection models.

## Implementation Details

The implementation of these metrics is simple and straightforward. That said, there are some guidelines for binary vs. multiple classes and more.

### Single Class

For classification, there are three types: **binary**, **multi-class,** and **multi-label**. In a binary classification task, TP, FN, FP, and TN are implemented as follows:

- `label_true` is the list of the ground truth label
- `label_pred` is the list of the predicted label or a label with a probability score of this sample belonging to the label
- `threshold` is a number that is compared against the prediction’s confidence score to decide whether the prediction is positive (when the score is greater or equal to `threshold`) or negative (otherwise)

Then the metric is defined as:

```python
TP = sum(true and pred for true, pred in zip(label_true, label_pred))
FN = sum(true and not pred for true, pred in zip(label_true, label_pred))
FP = sum(not true and pred for true, pred in zip(label_true, label_pred))
TN = sum(not true and not pred for true, pred in zip(label_true, label_pred)
```

Or when prediction scores are present:

```python
TP = sum(true and pred >= threshold for true, pred in zip(label_true, label_pred))
FN = sum(true and pred < threshold for true, pred in zip(label_true, label_pred))
FP = sum(not true and pred >= threshold for true, pred in zip(label_true, label_pred))
TN = sum(not true and pred < threshold for true, pred in zip(label_true, label_pred)
```

### Multiple Classes

Binary classification is pretty straightforward in handling these four metrics, but they’re handled differently for **multi-class** and **multi-label** classification tasks.

For any classification tasks with **multiple** classes, these four metrics are defined **per class**. This technique, also known as **one-vs-the-rest** (OVR), essentially evaluates each class as a binary classification problem.

In a **multi-label** classification task, a sample is considered to be a positive one if the ground truth **includes** the evaluating class; otherwise, it’s a negative sample. The same logic can be applied to the predictions, so, for example, if a classifier predicts that this sample belongs to class Airplane and Boat, and the ground truth for the same sample includes class Airplane, then this sample is considered to be a TP for class Airplane.

**Example of binary classification**

```python
>>> label_true = [0, 1, 0, 0, 1]
>>> label_pred = [0, 0, 1, 0, 1]
>>> print(f"{evaluate(label_true, label_pred)}")
TP=1
FN=1
FP=1
TN=2
```

**Example of multi-class classification**

```python
>>> label_true = [0, 1, 2, 1, 2]
>>> label_pred = [0, 2, 1, 1, 2]
>>> print(f"{evaluate(label_true, label_pred, labels=[0, 1, 2])}")
label 0:
TP=1
FN=0
FP=0
TN=4
label 1:
TP=1
FN=1
FP=1
TN=2
label 2:
TP=1
FN=1
FP=1
TN=2
```

**Example of multi-label classification**

```python
>>> label_true = [[0, 1, 1], [1, 0, 1], [1, 0, 0], [0, 0, 1]] # binarized
>>> label_pred = [[1, 0, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1]] # binarized
>>> print(f"{evaluate(label_true, label_pred, labels=[0, 1, 2])}")
label 0:
TP=2
FN=0
FP=2
TN=0
label 1:
TP=0
FN=1
FP=0
TN=3
label 2:
TP=2
FN=1
FP=1
TN=0
```

### Object Detection

There are some differences in how these four metrics work for a detection task compared to a classification task as they won’t be at a sample level but at an instance level that the model is detecting. For example, given an image with multiple objects, each prediction/ground truth is assigned to one group, and the definitions of the terms are slightly altered:

| TP | The correct detection |
| --- | --- |
| FP | The incorrect detection |
| FN | The missed ground truth |
| TN | The background region is correctly not detected (this metric is not used in a detection task because such regions are not explicitly annotated when labeling data) |

In an object detection task, checking for detection correctness requires a couple of other metrics (e.g., Intersection Over Union (IoU) and Geometry Matcher).

Let’s assume that a matching algorithm has already been run on all predictions and that the matched pairs and unmatched ground truths and predictions are given. Consider the following notations:

- `matched` is the list of **matched** ground truth and prediction bounding box **pairs** (from Geometry Matcher)
- `unmatched_gt` is the list of **unmatched** **ground truth** bounding boxes (from Geometry Matcher)
- `unmatched_pred` is the list of **unmatched** **prediction** bounding boxes (from Geometry Matcher)
- `threshold` is the threshold used to filter valid prediction bounding boxes based on their confidence scores

Then these metrics are defined as:

```python
TP = len([m.pred.confidence >= threshold for m in matched])
FN = len([m.pred.confidence < threshold for m in matched] + unmatched_gt)
FP = len([m.pred.confidence >= threshold for m in unmatched_pred])
```

Other than the differences that were mentioned above, the handling of single and multiple classes still applies to an object detection task.

Let’s look at some examples for different object detection tasks. Building on top of the examples previously used in the Geometry Matcher guide, these four metrics are computed given the ground truth/prediction match results:

**Example of single-class object detection**

A simple example of two ground truths and two predictions is one with a matched pair, one unmatched ground truth, and one unmatched prediction. Let’s see what the counts of TP, FP, and FN look like on this one:

![Legends](../assets/images/metrics-tpfpfntn-legends.png)

![Single-class example](../assets/images/metrics-tpfpfntn-single-class.png)

```python
>>> iou_threshold = 0.5
>>> score_threshold = 0.5
>>> matches = match_inferences([A, B], [a, b], iou_threshold)
>>> print(f"matches: {matches}")
matches: matched=[(A, a)], unmatched_gt=[B], unmatched_inf=[b]
>>> print(f"{evaluate(matches, score_threshold=0.5)}")
TP=1
FN=1
FP=1
```

**Example of multi-class object detection**

Similar to the multi-class classification, the four metrics are computed for class Apple and class Banana in the example below:

![Legends](../assets/images/metrics-tpfpfntn-legends.png)

![Multi-class example](../assets/images/metrics-tpfpfntn-multi-class.png)

```python
>>> iou_threshold = 0.5
>>> score_threshold = 0.5
>>> matches = match_inferences([A], [a, b], iou_threshold)
>>> print(f"matches: {matches}")
matches: matched=[], unmatched_gt=[A], unmatched_inf=[a, b]
>>> results = evaluate(matches, labels=["Apple", "Banana"], score_threshold=[0.5, 0.5])
>>> print(f"{results}")
class Apple:
TP=0
FN=1
FP=0
class Banana:
TP=0
FN=0
FP=1
```

### Aggregating Per-class Metrics

If you are looking for a **single** score that summarizes model performance across all classes, there are four different ways to aggregate per-class metrics: **macro**, **micro**, **weighted,** and **samples**. You can read more on these different averaging methods in [this guide](./averaging-methods.md).

## Limitations and Biases

TP, FP, TN, and FN are four metrics based on the assumption that each sample/instance can be classified as a positive or a negative, thus they can only be applied to single-class applications. The workaround for multiple-class applications is to compute these metrics for each label and then treat it as a single-class problem.

These four metrics are great at quantifying how well a model makes a correct prediction, but they don’t quantify how correct or incorrect each prediction is. A prediction with a score of **0.66**, for example, can be considered as a correct prediction as much as one with a score of **0.99**, if the score threshold is set to 0.5. As long as the confidence score is above the threshold, it is considered to be a positive prediction. To take a closer look at how correct the model is at predicting, you should consider plotting the score distribution histogram.
