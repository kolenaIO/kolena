---
icon: kolena/metrics-glossary-16

---

# :kolena-metrics-glossary-20: Task Metrics
Kolena supports automatic calculation of a number of metrics commonly used in machine learning tasks.

## Regression
Following commonly used regression metrics are supported out-of-the-box: 

1. [Mean Absolute Error](../../metrics/mean-absolute-error.md)
2. [Mean Squared Error](../../metrics/mean-squared-error.md)
3. [Root Mean Squared Error](../../metrics/root-mean-squared-error.md)
4. [R^2](../../metrics/coefficient-of-determination.md)
5. Pearson Correlation
6. Spearman Correlation

Follow the steps below to setup Regression metrics: 
<figure markdown>
![Defining Metrics](../../assets/images/task-metrics-regression.gif)
<figcaption>Task Metrics: Regression</figcaption>
</figure>

## Binary Classification
Following commonly used Binary classification metrics are supported out-of-the-box: 

1. [Accuracy](../../metrics/accuracy.md)
2. [Precision](../../metrics/precision.md)
3. [Recall](../../metrics/recall.md)
4. [F1 Score](../../metrics/f1-score.md)

Follow the steps below to setup Binary Classification metrics: 
<figure markdown>
![Defining Metrics](../../assets/images/task-metrics-binary-classification.gif)
<figcaption>Task Metrics: Binary classification</figcaption>
</figure>

## Multiclass Classification
Following commonly used multiclass classification metrics are supported out-of-the-box: 

1. [Accuracy](../../metrics/accuracy.md)
2. [Precision](../../metrics/precision.md)
3. [Recall](../../metrics/recall.md)
4. [F1 Score](../../metrics/f1-score.md)

You are able apply averaging methods ([macro](../../metrics/averaging-methods.md#macro-average), [micro](../../metrics/averaging-methods.md#micro-average) and [weighted](../../metrics/averaging-methods.md#weighted-average)) to above metrics.  

<figure markdown>
![Defining Metrics](../../assets/images/task-metrics-multiclass-classification.gif)
<figcaption>Task Metrics: Multiclass classification</figcaption>
</figure>

## Object detection
Following commonly used object detection metrics are supported out-of-the-box: 

1. [Average Precision](../../metrics/average-precision.md)
2. [Precision](../../metrics/precision.md)
3. [Recall](../../metrics/recall.md)
4. [F1 Score](../../metrics/f1-score.md)

??? "Required fields"
    Kolena attempts to automatically detect fields for true positive, false positive and false negative counts. For more information, please visit [Formatting results for Object Detection](../formatting-your-datasets.md#formatting-results-for-object-detection)


<figure markdown>
![Object detection](../../assets/images/task-metrics-object-detection.gif)
<figcaption>Task Metrics: Object detection</figcaption>
</figure>

## Custom Metrics
Kolena provides out-of-the-box aggregation options for your datapoint level evaluations that correspond with your desired metrics. For numeric evaluations you are able to select from count, mean, median, min, max, stddev and sum aggregations options. 
For categorical evaluations (class lable, boolean, etc) rate and count aggregation options are available.

<figure markdown>
    ![Numeric aggregation options](../../assets/images/numeric-aggregation-optins.png)
    ![Numeric aggregation](../../assets/images/categorical-aggregation-options.png)
</figure>

<div class="grid cards" markdown>
    
    ![Numeric aggregation options](../../assets/images/numeric-aggregation-optins.png)

    ![Numeric aggregation](../../assets/images/categorical-aggregation-options.png)
    
</dive>

