# Object Detection

## Overview of Object Detection

Object detection is a widely applied computer vision task. Object detection aims to locate and identify objects within
an image or video, typically with bounding boxes.

<figure markdown>
  ![COCO_val2014_000000023294](../assets/images/metrics-tasks-od-COCO_val2014_000000023294_annotated.png)
  <figcaption markdown>Person detection on an [MS COCO](https://paperswithcode.com/dataset/coco) image</figcaption>
</figure>

Object detection models require large amounts of annotated data to learn about objects of interest. Some commonly used
datasets are: [MS COCO](https://paperswithcode.com/dataset/coco),
[Pascal VOC](https://paperswithcode.com/dataset/pascal-voc),
[Open Images](https://paperswithcode.com/dataset/open-images-v7),
and [ImageNet](https://paperswithcode.com/dataset/imagenet).

Object detection has a wide range of applications from self-driving cars, facial recognition, video surveillance,
medical imaging, robotics, and more. Therefore, it is crucial to evaluate object detection models to understand their
performance and applicability any real-world situation.

## Evaluation Metrics for Object Detection Models

Evaluation of object detection models requires ground truths and model inferences. The ground truths in an image are
objects outlined by bounding boxes each labelled with a class, typically annotated by human labellers. The model
inferences for an image are the labelled bounding boxes having confidence scores (the model's certainty of
correctness), as if the model annotated the image.

A [bounding box matcher](./geometry-matching.md) can align an image's ground truths and model inferences to produce
[TP / FP / FN counts](./tp-fp-fn-tn.md). These counts are foundational for computing more detailed metrics, which
provide insight into the model performance. Different evaluation configurations can be tuned to filter out model
inferences before/after employing a matching algorithm such as filtering out inferences with a confidence score under
0.01 and/or ignoring matches where the [IoU](./iou.md) (the overlap between the ground truth and and inference
bounding box) under 0.5.

Below are some of the commonly used metrics for evaluating object detection models:

1. [**Precision**](./precision.md): Precision measures the ratio of correctly detected objects to all objects detected
    by the model. High precision indicates a low rate of false positives.

    $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

2. [**Recall**](./recall.md): Recall measures the ratio of correctly detected objects to all actual objects (the
    annotations). High recall indicates that the model is good at detecting most of the objects labelled by humans.

    $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

3. [**F1-Score**](./f1-score.md): F1-Score is the harmonic mean of precision and recall - a balance of both metrics as
    one metric.

    $$
    \begin{align}
    \text{F}_1 &= \frac {2} {\frac {1} {\text{Precision}} + \frac {1} {\text{Recall}}} \\[1em]
    &= \frac {2 \times \text{Precision} \times \text{Recall}} {\text{Precision} + \text{Recall}}
    \end{align}
    $$

4. [**Mean Average Precision (mAP)**](./averaging-methods.md): Mean average precision (mAP) is obtained by first
    computing the [average precision (AP)](./average-precision.md) for each class based on
    [Precision-Recall (PR) curves](./pr-curve.md) and then macro-averaging those scores across all classes. mAP is a
    comprehensive indicator of a model's performance across multiple categories.

    $$
    \begin{align}
    \text{mAP} = \frac{1}{N} \sum_{\text{class}}^{\text{all classes}} \text{AP}_\text{class}
    \end{align}
    $$

    !!! info "Guide: Averaging Methods"

        Read the [averaging methods](./averaging-methods.md) guide if you're not familiar with "macro" and "micro"
        terminology.

## Evaluation Plots for Object Detection Models

There are several common plots used to analyze the performance of object detection models.

??? info "Advanced Plot Insights"

    Suppose we are doing `person object detection` like within the first image above. It would be interesting to see
    how performance differs by different test cases or different characteristics of data. With different lines on
    plots representing different groups, such as `body part`, `race`, `bounding box size`, `image brightness`, etc.
    plots easily explain where performance drops.

    <center>
    ![pr_by_testcase.png](../assets/images/metrics-tasks-od-prcurve-light.png#only-light)
    ![pr_by_testcase.png](../assets/images/metrics-tasks-od-prcurve-dark.png#only-dark)
    </center>

    From the plot above, we see how a particular model suffers in performance when only a person's `arm` is shown.

1. [**Precision Recall (PR) Curves**](./pr-curve.md): The PR curve plots precision and recall within a unit square.
    The greater the area under the curve, the more performant your model is. Typically, there is a tradeoff between
    precision and recall, which you can read in detail [here](./pr-curve.md).

    <center>
    ![pr.png](../assets/images/metrics-prcurve-light.png#only-light)
    ![pr.png](../assets/images/metrics-prcurve-dark.png#only-dark)
    </center>

2. [**Confusion Matrix**](./confusion-matrix.md): Particularly for multiclass object detection, a confusion matrix
    displays the actual classes against the predicted classes in a table. This allows for a clear understanding of
    the model's performance with respect to classifying classes.

    <center>
    ![pr.png](../assets/images/metrics-confusion-matrix-light.png#only-light)
    ![pr.png](../assets/images/metrics-confusion-matrix-dark.png#only-dark)
    </center>

    From the plot above, we can see if the model is confused when detecting for cats and dogs.

Note that any plot suitable for a classification task is also suitable for an object detection task, as object
detection is a combination of a classification and localization task. However, it is important to consider what
matters to your project and design custom plots to describe your model's performance in the best way possible.
For example: an F1-Score vs confidence threshold plot, a histogram of IoUs, or mAP across different IoUs.

In conclusion, evaluating object detection models is a multifaceted process based on the comparison ground truths
and model inferences. Depending on your own project needs, custom metrics and plots should be considered.
Understanding and effectively applying these evaluation metrics are essential for optimizing object detection
models to meet the demands of real-world applications.
