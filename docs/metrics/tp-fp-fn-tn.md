# TP / FP / FN / TN

The counts of **true positive** (TP), **false positive** (FP), **false negative** (FN), and **true negative** (TN)
predictions and ground truths are essential for summarizing model performance. These metrics are the building blocks of
many other metrics, including accuracy, precision, and recall.

| Metric | | Description |
| --- | --- | --- |
| **True Positive** | TP | An instance for which both predicted and actual values are positive |
| **False Positive** | FP | An instance for which predicted value is positive but actual value is negative |
| **<nobr>False Negative</nobr>** | FN | An instance for which predicted value is negative but actual value is positive |
| **True Negative** | TN | An instance for which both predicted and actual values are negative |

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

    ```python
    >>> ground_truths = [False, True, False, False, True]
    >>> inferences = [0.3, 0.2, 0.9, 0.4, 0.5]
    >>> T = 0.5
    >>> TP = sum(    gt and inf >= T for gt, inf in zip(ground_truths, inferences))
    >>> FP = sum(not gt and inf >= T for gt, inf in zip(ground_truths, inferences))
    >>> FN = sum(    gt and inf <  T for gt, inf in zip(ground_truths, inferences))
    >>> TN = sum(not gt and inf <  T for gt, inf in zip(ground_truths, inferences))
    >>> print(f"TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    TP=1, FN=1, FP=1, TN=2
    ```


#### Multiclass

TP / FP / FN / TN metrics are computed a little differently in **multiclass** classification tasks.

For **multiclass** classification tasks, these four metrics are defined **per class**. This technique,
also known as **one-vs-rest** (OvR), essentially evaluates each class as a binary classification problem.

Consider a classification problem where a given image belongs to either the `Airplane`, `Boat`, or `Car` class. Each of
these TP / FP / FN / TN metrics is computed for each class. For class `Airplane`, the metrics are defined as follows:

| Metric | Example |
| --- | --- |
| **True Positive** | Any image predicted as an `Airplane` that is labeled as an `Airplane` |
| **False Positive** | Any image predicted as an `Airplane` that is _not_ labeled as an `Airplane` (e.g. labeled as `Boat` but predicted as `Airplane`) |
| **<nobr>False Negative</nobr>** | Any image _not_ predicted as an `Airplane` that is labeled as an `Airplane` (e.g. labeled as `Airplane` but predicted as `Car`) |
| **True Negative** | Any image _not_ predicted as an `Airplane` that is _not_ labeled as an `Airplane` (e.g. labeled as `Boat` but predicted as `Boat` or `Car`) |

#### Multi-label

In a **multi-label** classification task, TP / FP / FN / TN are computed per class, like in multiclass classification.

A sample is considered to be a positive one if the ground truth **includes** the evaluating class; otherwise, it’s a
negative sample. The same logic can be applied to the predictions, so, for example, if a classifier predicts that this
sample belongs to class `Airplane` and `Boat`, and the ground truth for the same sample is only class `Airplane`, then
this sample is considered to be a TP for class `Airplane`, and FP for class `Boat`.

Multi-label classification tasks can alternately be thought of as a collection of binary classification tasks.

### Object Detection

There are some differences in how these four metrics work for a detection task compared to a classification task as they won’t be at a sample level but at an instance level that the model is detecting. For example, given an image with multiple objects, each prediction/ground truth is assigned to one group, and the definitions of the terms are slightly altered:

| Metric | | Description |
| --- | --- | --- |
| **True Positive** | TP | Inference matched with a ground truth |
| **False Positive** | FP | Inference that is not matched with a ground truth |
| **<nobr>False Negative</nobr>** | FN | Ground truth that is not matched with an inference |
| **True Negative** | TN | <p>Poorly defined for object detection.</p><div>One common definition of true negative for object detection is any image that does not have any true positive or false positive inferences.</div> |

In an object detection task, checking for detection correctness requires a couple of other metrics (e.g., Intersection
over Union (IoU) and Geometry Matching).

Let’s assume that a matching algorithm has already been run on all predictions and that the matched pairs and unmatched
ground truths and predictions are given. Consider the following notations:

- `matched`: is the list of **matched** ground truth and prediction bounding box **pairs** (from Geometry Matcher)
- `unmatched_gt`: is the list of **unmatched** ground truth bounding boxes (from Geometry Matcher)
- `unmatched_inf`: is the list of **unmatched** inference bounding boxes (from Geometry Matcher)
- `T`: the threshold used to filter valid prediction bounding boxes based on their confidence scores

Then these metrics are defined:

```python
TP = len([m.pred.confidence >= T for m in matched])
FN = len([m.pred.confidence <  T for m in matched] + unmatched_gt)
FP = len([m.pred.confidence >= T for m in unmatched_inf])
```

Other than the differences that were mentioned above, the handling of single and multiple classes still applies to an
object detection task.

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

### Averaging Per-class Metrics

For problems with multiple classes, these TP / FP / FN / TN metrics are computed for each class. If you are looking for
a single score that summarizes model performance across all classes, there are a few different ways to aggregate
per-class metrics: **macro**, **micro**, and **weighted.**

Read more about these different averaging methods in the [Averaging Methods guide](./averaging-methods.md).

## Limitations and Biases

TP, FP, TN, and FN are four metrics based on the assumption that each sample/instance can be classified as a positive
or a negative, thus they can only be applied to single-class applications. The workaround for multiple-class
applications is to compute these metrics for each label and then treat it as a single-class problem.

These four metrics are great at quantifying how well a model makes a correct prediction, but they don’t quantify model
confidence
correct or incorrect each prediction is. A prediction with a score of **0.66**, for example, can be considered as a
correct prediction as much as one with a score of **0.99**, if the score threshold is set to 0.5. As long as the
confidence score is above the threshold, it is considered to be a positive prediction. To take a closer look at how
correct the model is at predicting, you should consider plotting the score distribution histogram.
