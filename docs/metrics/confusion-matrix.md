# Confusion Matrix

## Description

A confusion matrix is a structured plot describing machine learning model performance as a table that highlights counts of objects with predicted classes (columns) against the actual classes (rows). Each cell has a count of the number of objects that have its correct class and predicted class, which indicates how confused a model is while labeling objects.

A model is confused when a predicted class does not match the actual class. When they do match, this is considered a true positive (TP). You can find more info on TPs, true negatives (TNs), false positives (FPs), and false negatives (FNs) here: [Metric :: TP / FP / FN / TN](./tp-fp-fn-tn.md). In general, a model resulting in more TPs/TNs with fewer FPs/FNs is better.

## Intended Uses

Confusion matrices are used in classification workflows with only one class or with multiple classes, which extends to object detection workflows, too. They help evaluate models by counting errors, visualizing class imbalances, and model comparison when comparing confusion matrices of different models.

## Implementation Details

The implementation of a confusion matrix depends on whether the workflow concerns one or more classes.

### Single-Class Implementation

Single-class confusion matrices are similar to binary classification problems. After computing the number of TPs, FPs, FNs, and TNs, a confusion matrix would look like this:

|  | Predicted Positive | Predicted Negative |
| --- | --- | --- |
| Actual Positive | True Positive (TP) | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN) |

**Example 1**

```python
labels = ['cat', 'cat', 'cat', 'no cat', 'no cat']
predictions = ['cat', 'no cat', 'no cat', 'no cat', 'cat']
```

|  | Predicted Cat | Predicted No Cat |
| --- | --- | --- |
| Cat | 1 | 2 |
| No Cat | 1 | 1 |

### Multi-class Implementation

Multi-class confusion matrices outline counts of TPs, FPs, and FNs for every unique pair of actual and predicted labels:

```python
def create_multiclass_confusion_matrix(labels, predictions):
    # Get unique labels
    unique_labels = list(set(labels + predictions))
    unique_labels.sort()

    # Initialize confusion matrix
    confusion_matrix = [[0] * len(unique_labels) for _ in range(len(unique_labels))]

    # Populate confusion matrix
    for label, prediction in zip(labels, predictions):
        label_idx = unique_labels.index(label)
        pred_idx = unique_labels.index(prediction)
        confusion_matrix[label_idx][pred_idx] += 1

    return confusion_matrix
```

A multi-class classification confusion matrix with four classes (observing class B) would have the following format:

|  | Predicted Class A | Predicted Class B | Predicted Class C | Predicted Class D |
| --- | --- | --- | --- | --- |
| Actual Class A | True Negative | False Positive | True Negative | True Negative |
| Actual Class B | False Negative | True Positive | False Negative | False Negative |
| Actual Class C | True Negative | False Positive | True Negative | True Negative |
| Actual Class D | True Negative | False Positive | True Negative | True Negative |

**Example 2**

```python
labels = ['cat', 'dog', 'fish', 'cat', 'dog', 'dog']
predictions = ['cat', 'dog', 'cat', 'cat', 'dog', 'fish']
```

|  | Predicted Cat | Predicted Dog | Predicted Fish |
| --- | --- | --- | --- |
| Cat | 2 | 0 | 0 |
| Dog | 0 | 2 | 1 |
| Fish | 1 | 0 | 0 |

**Example 3**

In a different case, these counts may be much higher:

|  | Predicted Cat | Predicted Dog | Predicted Fish |
| --- | --- | --- | --- |
| Cat | 200 | 0 | 0 |
| Dog | 100 | 8,800 | 600 |
| Fish | 100 | 0 | 1000 |

This confusion matrix reveals that a model is very good at identifying dogs: 8,800 of 9,500 dogs were predicted to be dogs. Of the 700 missing dog predictions, 600 were predicted to be fish, and 100 were predicted to be cats.

This confusion matrix indicates that when a model makes a “cat” inference, the model is correct half the time. If it is incorrect, it has labeled the “cat” as a “dog” or “fish.”

Whenever there is an actual “cat” class, the model never predicts that there is a different animal.

### Other Implementations

Sometimes it is easier to see a normalized confusion matrix when you want to focus on class-level behavior. If confusion matrices are color-coded, normalizing can also create a better visual representation:

![Example of normalized and colored confusion matrix](../assets/images/metrics-confusion-matrix-normalized.png)

You can normalize a confusion matrix by `row` (actual classes), `column` (predicted classes), or `all` (entire matrix). Each type of normalization surfaces a view sharing different information, which is outlined below.

**Normalizing by `row`**
For an actual class, this normalization allows us to see the proportion of correctly or incorrectly predicted objects for each predicted class. To normalize by row, divide each entry in that row by the sum of values within it. If we normalize Example 3 by `row`, we get:

|  | Predicted Cat | Predicted Dog | Predicted Fish |
| --- | --- | --- | --- |
| Cat | 1 | 0 | 0 |
| Dog | 0.01 | 0.93 | 0.06 |
| Fish | 0.09 | 0 | 0.91 |

**Normalizing by `column`**
For a predicted class, this normalization allows us to see the proportion of instances predicted as a certain class that actually belong to each true class. To normalize by column, divide each entry in a column by the sum of values within that column. If we normalize Example 3 by `column`, we get:

|  | Predicted Cat | Predicted Dog | Predicted Fish |
| --- | --- | --- | --- |
| Cat | 0.5 | 0 | 0 |
| Dog | 0.25 | 1 | 0.375 |
| Fish | 0.25 | 0 | 0.625 |

**Normalizing by `all`**
For each entry, this normalization allows us to see the overall proportion of instances that fall into a combination of an actual and predicted class. To normalize by all, divide each entry by the total sum of all the values in the matrix. If we normalize Example 3 by `all`, we get:

|  | Predicted Cat | Predicted Dog | Predicted Fish |
| --- | --- | --- | --- |
| Cat | 0.02 | 0 | 0 |
| Dog | 0.01 | 0.81 | 0.06 |
| Fish | 0.01 | 0 | 0.09 |

## Limitations and Biases

Confusion matrices are great for the evaluation of models that deal with multiple classes. They are structured tables of numbers, which is its strength and weakness.

1. Class imbalance: Confusion matrices can appear biased when dealing with imbalanced numbers of instances per class, leading to skewed numbers. This can be addressed by normalizing the matrix.
2. Categorical evaluation: Confusion matrices have categorical outputs and do not surface any details for misclassifications. All misclassifications are treated equally, so there may be cases where classes are similar or hierarchically related, but confusion matrices will not account for these details.
