# Object Detection

## Overview of Object Detection

Object detection is a widely applied computer vision task. Object detection aims to locate and identify objects within
an image or video, typically with bounding boxes.

<figure markdown>
  ![COCO Multiclass OD](../assets/images/metrics-tasks-od-multiclass.png)
  <figcaption markdown>Vehicle detection on an [MS COCO](https://paperswithcode.com/dataset/coco) image</figcaption>
</figure>

Object detection models require large amounts of annotated data to learn about objects of interest. Some commonly used
datasets are: [MS COCO](https://paperswithcode.com/dataset/coco),
[Pascal VOC](https://paperswithcode.com/dataset/pascal-voc),
[Open Images](https://paperswithcode.com/dataset/open-images-v7),
and [ImageNet](https://paperswithcode.com/dataset/imagenet).

Object detection has a wide range of applications from self-driving cars, facial recognition, video surveillance,
medical imaging, robotics, and more. Therefore, it is crucial to evaluate object detection models to understand their
performance and applicability in any real-world situation.

## Evaluation Metrics for Object Detection Models

Evaluation of object detection models requires ground truths and model inferences. The ground truths in an image are
objects outlined by bounding boxes each labeled with a class, typically annotated by human labelers. The model
inferences for an image are the labeled bounding boxes having confidence scores (the model's certainty of
correctness), as if the model annotated the image.

A [bounding box matcher](./geometry-matching.md) can align an image's ground truths and model inferences to produce
[TP / FP / FN counts](./tp-fp-fn-tn.md). These counts are fundamental for computing more detailed metrics, which
provide insight into the model's performance. Different evaluation configurations can be tuned to filter out model
inferences before/after employing a matching algorithm such as filtering out inferences with a confidence score under
0.01 and/or ignoring matches where the [IoU](./iou.md) (the overlap between the ground truth and inference
bounding box) under 0.5.

Below are some of the commonly used metrics for evaluating object detection models:

1. [**Precision**](./precision.md): Precision measures the ratio of correctly detected objects to all objects detected
    by the model. High precision indicates a low rate of false positives.

    $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

2. [**Recall**](./recall.md): Recall measures the ratio of correctly detected objects to all actual objects (the
    ground truths). High recall indicates that the model is good at detecting most of the objects labeled by humans.

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

        Read the [averaging methods](./averaging-methods.md) guide if you are not familiar with "macro" and "micro"
        terminology.

## Evaluation Plots for Object Detection Models

There are several common plots used to analyze the performance of object detection models.

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

    From the plot above, we can see if the model is confused when detecting cats and dogs.

??? info "Advanced Plot Insights"

    For `person object detection`, it would be interesting to see
    how performance differs by different test cases or different characteristics of data. With different lines on
    plots representing different groups, such as `body part`, `race`, `bounding box size`, `image brightness`, etc.
    plots easily explain where performance drops.

    <center>
    ![pr_by_testcase.png](../assets/images/metrics-tasks-od-prcurve-light.png#only-light)
    ![pr_by_testcase.png](../assets/images/metrics-tasks-od-prcurve-dark.png#only-dark)
    </center>

    From the example above, we see how a particular model suffers in performance when only a person's `arm` is shown.

Note that any plot suitable for a classification task is also suitable for an object detection task, as object
detection is a combination of a classification and localization task. However, it is important to consider what
matters to your project and design custom plots to describe your model's performance in the best way possible.
For example: an F1-Score vs confidence threshold plot, a histogram of IoUs, or mAP vs different IoUs.

## Using Kolena to Evaluate Object Detection Models

Kolena can make object detection model evaluation very easy. Upload your dataset of datapoints (e.g. locators to
images) with ground truth [annotations](../dataset/formatting-your-datasets.md#kolena-annotations) (e.g. labeled
bounding boxes) by importing it directly from the user interface or using the
[SDK](../dataset/quickstart.md/#step-1-upload-dataset).

Suppose we have a dataset with multiple images, containing an image that exists at
`s3://my-images/intersection_1.png`. In that image, we have two ground truth bounding boxes expressed as a
[`List[LabeledBoundingBox]`](../reference/annotation.md#kolena.annotation.LabeledBoundingBox):
```python
from kolena.annotation import LabeledBoundingBox
bboxes = [
    LabeledBoundingBox(top_left=(538.03, 8.86), bottom_right=(636.85, 101.93), label="car"),
    LabeledBoundingBox(top_left=(313.02, 12.01), bottom_right=(553.98, 99.84), label="truck"),
]
```
Or equivalently, as a JSON string on one line (shown with multiple lines for formatting):
```
"[{""top_left"": [538.03, 8.86], ""bottom_right"": [636.85, 101.93],
   ""label"": ""car"", ""data_type"": ""ANNOTATION/BOUNDING_BOX""},
  {""top_left"": [313.02, 12.01], ""bottom_right"": [553.98, 99.84],
   ""label"": ""truck"", ""data_type"": ""ANNOTATION/BOUNDING_BOX""}]"
```
The expected CSV format for this dataset should look like the CSV below:
```
locator,image_id,ground_truths,extra_information_as_metadata
s3://my-images/intersection_1.png,1,"[{""top_left"": [538.03, 8.86], ... }, {""top_left"": ...}]",cloudy
s3://my-images/intersection_2.png,2,"[{""top_left"": [1380.35, 4.84], ... }]",rainy
...
```
This CSV can be directly uploaded to Kolena's platform. To upload a dataset with the SDK, see details for
[structured data usage](../dataset/formatting-your-datasets.md/#structured-data).

Once your dataset and ground truths are uploaded, you can upload model inferences and results in a similar fashion.
An object detection model will predict the bounding boxes of objects within each image, so there is a list of
inferences per image:
[`List[ScoredLabeledBoundingBox]`](../reference/annotation.md#kolena.annotation.ScoredLabeledBoundingBox).

<div class="grid" markdown>
<figure markdown>
  ![OD_metrics.png](../assets/images/metrics-tasks-od-qs-light.png#only-light)
  ![OD_metrics.png](../assets/images/metrics-tasks-od-qs-dark.png#only-dark)
  <figcaption markdown>Configurable metrics for custom [Quality Standards](../dataset/core-concepts/quality-standard.md)
  </figcaption>
</figure>

!!! info "Built-In Object Detection Metrics"

    Rather than only uploading raw inferences, it is also worthwhile to upload metrics such as
    [TP / FP / FN](./tp-fp-fn-tn.md) from your own [bounding box matcher](./geometry-matching.md) as counts and as
    lists of annotations so that they appear in their respective groups within the Studio.

    Furthermore, Kolena can automatically compute common object detection metrics such as
    [precision](./precision.md) and [recall](./recall.md), making it easy to visualize an assortment of plots
    on the fly.

    ![OD_metrics.png](../assets/images/metrics-tasks-od-debugger-light.png#only-light)
    ![OD_metrics.png](../assets/images/metrics-tasks-od-debugger-dark.png#only-dark)

    In the debugger, you are able to see plots by class or by test case as shown above. For more details on
    automatic metrics and plots, please refer to documentation for
    [formatting results](../dataset/formatting-your-datasets.md/#formatting-results-for-object-detection).

</div>

Note that a [`ScoredLabeledBoundingBox`](../reference/annotation.md#kolena.annotation.ScoredLabeledBoundingBox)
has an extra `score` attribute for the confidence of the model:
```
{""top_left"": ... ""label"": ""car"", ""score"": 0.94, ""data_type"":""ANNOTATION/BOUNDING_BOX""}
```

The expected CSV format for model results should look like the CSV below:
```
image_id,count_TP,count_FP,count_FN,TP,FP,FN,custom_information
1,2,0,0,"[{""top_left"": [538.5, 9.0], ... }, {""top_left"": ... }]","[]","[]",0.88
2,0,0,1,"[]","[]","[{""top_left"": ... }]",0.5
...
```

By using `image_id` as the unique ID for images, we can join the model results in the results CSV to the correct
datapoint in the dataset. Once annotations and model results are formatted for Kolena, comparing models across
different scenarios and metrics becomes as easy as uploading files -- one CSV upload for the dataset, and one CSV
upload for the model results.

In conclusion, evaluating object detection models is a multifaceted process based on the comparison of ground truths
and model inferences. Depending on your own project needs, custom metrics and plots should be considered.
Understanding and effectively applying evaluation metrics are essential for optimizing object detection
models to meet the demands of real-world applications.
