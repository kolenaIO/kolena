---
search:
  exclude: true
---

# Accuracy

Because it is simple to understand and easy to implement, accuracy is one of the most well-known metrics in machine learning model evaluation. It works by measuring how often a model correctly predicts something (ranging from 0 to 1, with 1 being a perfect prediction), and it serves as a ratio of the number of correct predictions to the total number of predictions, making it a good metric for assessing model performance in simple cases with balanced data. That said, it runs into some issues if the data is unbalanced.


## Implementation Details
Accuracy is generally used to train or evaluate classification models. Aside from classification, it has also been used to evaluate semantic segmentation models by measuring the percent of correctly classified pixels in an image.

In a classification task, accuracy is simply a ratio of the number of correct predictions to the total number of predictions.

Consider the following notation, where

- $y_{true}$ is the list of ground truths
- $y_{pred}$ is the list of predictions

The metric is then defined as

$$
accuracy(y_{true}, y_{pred}) =  \frac {TP + TN} {TP + TN + FP + FN}
$$

Read this [guide](./tp-fp-fn-tn.md) if you are not familiar with TP, FP, FN and TN.

Here are some examples of what accuracy looks like in different scenarios of classification tasks:

**Example of perfect predictions**

```python
>>> y_true = [0, 0, 0, 1, 1, 1]
>>> y_pred = [0, 0, 0, 1, 1, 1]
>>> print(f"accuracy: {accuracy(y_true, y_pred)}")
accuracy: 1.0
```

**Example of some incorrect predictions**

```python
>>> y_true = [0, 1, 2, 3]
>>> y_pred = [0, 2, 1, 3]
>>> print(f"accuracy: {accuracy(y_true, y_pred)}")
accuracy: 0.5
```

**Example of imbalanced data**

```python
>>> y_true = [0, 0, 0, 0, 0, 0, 1, 1]
>>> y_pred = [0, 0, 0, 0, 0, 0, 0, 0]
>>> print(f"accuracy: {accuracy(y_true, y_pred)}")
accuracy: 0.75
```

## Limitations and Biases

While accuracy generally describes a classifier’s performance (in most scenarios), it is important to note that the metric can be deceptive, especially when [the data is imbalanced](https://stephenallwright.com/imbalanced-data/). For example, let’s say there are a total of 500 samples, with 450 belonging to the positive class and 50 to the negative. If the model correctly predicts all the positive samples but misses all the negative ones, its accuracy is `450 / 500 = 0.9`.  But is the model truly 90% accurate when it fails to classify the negative samples 100% of the time? Using the accuracy metric alone can hide a model’s true performance, so we recommend other metrics that are better suited for imbalanced data: [balanced accuracy](https://stephenallwright.com/balanced-accuracy/), [precision](./precision.md), [recall](./recall.md), f-score, etc.
