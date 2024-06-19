---
icon: kolena/detection-20
---

# :kolena-detection-20: How to Test Object Detection Models

This guide outlines how to format your object detection data on Kolena, and walks through the steps to easily
test object detection models.

## Overview of Object Detection

<div class="grid" markdown>
<div markdown>
Object detection is a widely applied computer vision task. Object detection aims to locate and identify objects
within an image or video, typically with bounding boxes.
Object detection has a wide range of applications from self-driving cars, facial recognition,
video surveillance, medical imaging, robotics, and more. Therefore, it is crucial to evaluate object detection
models to understand their performance and applicability in any real-world situation.
</div>
<figure markdown>
![COCO Multiclass OD](../assets/images/task-od-multiclass.png)
<figcaption markdown>Vehicle detection on an [MS COCO](https://paperswithcode.com/dataset/coco) image</figcaption>
</figure>
</div>

??? info "Examples of Object Detection Datasets"

    Object detection models require large amounts of annotated data to learn about objects of interest.
    Some commonly used
    datasets are: [MS COCO](https://paperswithcode.com/dataset/coco),
    [Pascal VOC](https://paperswithcode.com/dataset/pascal-voc),
    [Open Images](https://paperswithcode.com/dataset/open-images-v7),
    and [ImageNet](https://paperswithcode.com/dataset/imagenet).

## Object Detection Model Evaluation

It is important to understand some core performance metrics and plots that are typically seen when
evaluating object detection models.

### Object Detection Metrics

Evaluation of object detection models requires ground truths and model inferences. The ground truths in an image are
objects outlined by bounding boxes each labeled with a class, typically annotated by human labelers. The model
inferences for an image are the labeled bounding boxes having confidence scores (the model's certainty of
correctness), as if the model annotated the image.

A [bounding box matcher](../metrics/geometry-matching.md) can align an image's ground truths and model inferences
to produce [TP / FP / FN counts](../metrics/tp-fp-fn-tn.md). These counts are fundamental for computing more
detailed metrics, which provide insight into the model's performance. Different evaluation configurations can be
tuned to filter out model inferences before/after employing a matching algorithm such as filtering out inferences
with a confidence score under 0.01 and/or ignoring matches where the [IoU](../metrics/iou.md) (the overlap
between the ground truth and inference bounding box) under 0.5.

1. [**Precision**](../metrics/precision.md): Precision measures the ratio of correctly detected objects to
    all objects detected by the model. High precision indicates a low rate of false positives.

    $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

2. [**Recall**](../metrics/recall.md): Recall measures the ratio of correctly detected objects to all actual objects
    (the ground truths). High recall indicates that the model is good at detecting most of the objects labeled
    by humans.

    $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

3. [**F1-Score**](../metrics/f1-score.md): F1-Score is the harmonic mean of precision and recall - a balance of both
    metrics as one metric.

    $$
    \begin{align}
    \text{F}_1 &= \frac {2} {\frac {1} {\text{Precision}} + \frac {1} {\text{Recall}}} \\[1em]
    &= \frac {2 \times \text{Precision} \times \text{Recall}} {\text{Precision} + \text{Recall}}
    \end{align}
    $$

4. [**Mean Average Precision (mAP)**](../metrics/averaging-methods.md): Mean average precision (mAP) is
    obtained by first computing the [average precision (AP)](../metrics/average-precision.md) for each class
    based on [Precision-Recall (PR) curves](../metrics/pr-curve.md) and then macro-averaging those scores
    across all classes. mAP is a comprehensive indicator of a model's performance across multiple categories.

    $$
    \begin{align}
    \text{mAP} = \frac{1}{N} \sum_{\text{class}}^{\text{all classes}} \text{AP}_\text{class}
    \end{align}
    $$

    Read the [averaging methods](../metrics/averaging-methods.md) guide if you are not familiar with "macro"
    and "micro" terminology.

### Object Detection Plots

Plots can become very powerful ways to gain insights into unexpected model behavior, and a formal way to showcase
strong model quality. There are several common plots used to analyze the performance of object detection models.

1. [**Precision Recall (PR) Curves**](../metrics/pr-curve.md): The PR curve plots precision and recall within a
    unit square. The greater the area under the curve, the more performant your model is. Typically, there is
    a tradeoff between precision and recall, which you can read in detail [here](../metrics/pr-curve.md).

    <center>
    ![pr.png](../assets/images/metrics-prcurve-light.png#only-light)
    ![pr.png](../assets/images/metrics-prcurve-dark.png#only-dark)
    </center>

2. [**Confusion Matrix**](../metrics/confusion-matrix.md): Particularly for multiclass object detection, a
    confusion matrix displays the actual classes against the predicted classes in a table. This allows for a
    clear understanding of the model's performance with respect to classifying classes.

    <center>
    ![pr.png](../assets/images/metrics-confusion-matrix-light.png#only-light)
    ![pr.png](../assets/images/metrics-confusion-matrix-dark.png#only-dark)
    </center>

    From the plot above, we can see if the model is confused when detecting cats and dogs.

Note that any plot suitable for a classification task is also suitable for an object detection task, as object
detection is a combination of a classification and localization task. However, it is important to consider what
matters to your project and design custom plots to describe your model's performance in the best way possible.
For example: an F1-Score vs confidence threshold plot, a histogram of IoUs, or mAP vs different IoUs.

??? info "Advanced Plot Insights"

    For `person object detection`, it would be interesting to see
    how performance differs by different test cases or different characteristics of data. With different lines on
    plots representing different groups, such as `body part`, `race`, `bounding box size`, `image brightness`, etc.
    plots easily explain where performance drops.

    <center>
    ![pr_by_testcase.png](../assets/images/task-od-prcurve-light.png#only-light)
    ![pr_by_testcase.png](../assets/images/task-od-prcurve-dark.png#only-dark)
    </center>

    From the example above, we see how a particular model suffers in performance when only a
    person's `arm` is shown.

## Using Kolena to Evaluate Object Detection Models

In this guide, we will use the [COCO 2014](https://cocodataset.org/#home) dataset to demonstrate test data curation
and model evaluation in Kolena for object detection, using both the web app and the `kolena` Python SDK.
In the examples below, using the web app will walk you through a single class object detection task,
and following the SDK instructions will demonstrate a multiclass object detection task, just for some variety.

Kolena makes object detection model evaluation very easy in just three simple steps:

### Step 1: Format your Object Detection Dataset

When preparing to upload your object detection dataset, ensure that it conforms to one of the
[supported file formats](/dataset/advanced-usage/formatting-your-datasets.md#supported-file-data-formats) to
guarantee compatibility with Kolena's data processing capabilities (`.csv`, `.parquet`, or `.jsonl`).
We will walk you through an example using CSVs below:

The bounding box annotations in the dataset have been defined using
[`List[LabeledBoundingBox]`](../reference/annotation.md#kolena.annotation.LabeledBoundingBox) from
the `kolena` SDK.

Suppose we have a dataset with multiple images, and one of the images exists at
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
At minimum, an object detection CSV should include a `locator` column as the ID field, and a `ground_truths`
column for the dataset's annotations. Additional metadata such as `image_id` or attributes about the weather
are useful but optional.

This CSV can be directly uploaded to Kolena's platform. To upload a dataset with the SDK, see details for
[structured data usage](../dataset/advanced-usage/formatting-your-datasets.md#structured-data).

See the [`object_detection_2d/upload_dataset.py`](https://github.com/kolenaIO/kolena/blob/trunk/examples/dataset/object_detection_2d/object_detection_2d/upload_dataset.py)
script in the code example for details on how a multiclass object detection dataset was generated and
an example for preparing a dataset file.

### Step 2: Upload your Object Detection Dataset

Model evaluations on Kolena starts with datasets. Upload your dataset of datapoints (e.g. locators to
images) with ground truth [annotations](../dataset/advanced-usage/formatting-your-datasets.md#kolena-annotations)
(e.g. labeled bounding boxes) by importing the dataset file directly within the web app or using the SDK.

=== "Web App"
    To upload a dataset, having a properly formatted dataset file is a prerequisite.

    To get started, navigate to [kolena.com](https://app.kolena.com/redirect/datasets) and
    click `Import Dataset` then `Select From Cloud Storage`.
    Using the explorer, navigate to `s3://kolena-public-examples/coco-2014-val/`
    and select `coco-2014-val_person-detection.csv`. This is a dataset file for person detection using the COCO
    dataset.

    You will now see a preview of how the information is going to be consumed by Kolena.

    Give your dataset a name and select `locator` as the ID field. The ID field uniquely identifies a datapoint
    and is used when uploading model results to associate results with datapoints.

    Click `Import` to create the dataset. Once the import has completed,
    you can add descriptions and tags to organize your datasets.

    <figure markdown>
    ![COCO Dataset Upload](../assets/images/task-od-make-dataset-light.gif#only-light)
    ![COCO Dataset Upload](../assets/images/task-od-make-dataset-dark.gif#only-dark)
    <figcaption markdown>COCO Dataset Upload</figcaption>
    </figure>

=== "SDK"

    The example code contains a script [`object_detection_2d/upload_dataset.py`](https://github.com/kolenaIO/kolena/blob/trunk/examples/dataset/object_detection_2d/object_detection_2d/upload_dataset.py)
    which will process the CSV file `s3://kolena-public-examples/coco-2014-val/transportation/raw/coco-2014-val.csv`
    and register a small transportation-based dataset in Kolena using the `register_dataset` function.

    First, let's first configure our environment by populating the `KOLENA_TOKEN`
    environment variable. Visit the
    [:kolena-developer-16: Developer](https://app.kolena.com/redirect/developer) page to
    generate an API token and copy and paste the code snippet into your environment:

    ```shell
    export KOLENA_TOKEN="********"
    ```

    We can now register a new dataset using the provided script:

    ```shell
    poetry run python3 object_detection_2d/upload_dataset.py
    ```

    After this script has completed, a new dataset named `coco-2014-val` will be created,
    and see in [:kolena-dataset-20: Datasets](https://app.kolena.com/redirect/datasets).

### Step 3: Upload Object Detection Model Results

For this example, we will upload object detection model results
of [YOLO X](https://github.com/Megvii-BaseDetection/YOLOX) for the single class object detection task in the web app,
and upload the results of [Faster R-CNN](https://github.com/facebookresearch/Detectron) for the
multiclass object detection task through the SDK.

??? info "Generating Model Results"

    An object detection model will predict the bounding boxes of objects within each image, so there is a list of
    inferences per image:
    [`List[ScoredLabeledBoundingBox]`](../reference/annotation.md#kolena.annotation.ScoredLabeledBoundingBox).
    Note that a [`ScoredLabeledBoundingBox`](../reference/annotation.md#kolena.annotation.ScoredLabeledBoundingBox)
    has an extra `score` attribute for the confidence of the model:
    ```
    {""top_left"": ... ""label"": ""car"", ""score"": 0.94, ""data_type"":""ANNOTATION/BOUNDING_BOX""}
    ```

    The expected CSV format for model results should look like the CSV below:
    ```
    locator,count_TP,count_FP,count_FN,TP,FP,FN,custom_information
    s3://my-images/intersection_1.png,2,0,0,"[{""top_left"": [538.5, 9.0], ... }, {""top_left"": ... }]","[]","[]",0.88
    s3://my-images/intersection_2.png,0,0,1,"[]","[]","[{""top_left"": ... }]",0.5
    ...
    ```

    By using `locator` as the unique identifier for images, Kolena can join the model results in the CSV to the correct
    datapoint in the dataset.

    See the [`object_detection_2d/upload_results.py`](https://github.com/kolenaIO/kolena/blob/trunk/examples/dataset/object_detection_2d/object_detection_2d/upload_results.py)
    script in the code example for details on how results were generated, which uses [`upload_object_detection_results`](../reference/pre-built/object-detection-2d/#kolena._experimental.object_detection.upload_object_detection_results)

=== "Web App"

    To upload new model results, from the `Details` tab of the dataset, click on `Upload Model Results`
    in the upper right.
    Then, select `Upload From Cloud Storage`. Using the explorer, navigate to
    `s3://kolena-public-examples/coco-2014-val/person-detection/results/` and select the `yolo_x` CSV.

    You will now see a preview of how Kolena will ingest the model results. Give your model a name,
    and click `Import` to upload the model results.

    <figure markdown>
    ![Model Results Upload](../assets/images/task-od-upload-results-light.gif#only-light)
    ![Model Results Upload](../assets/images/task-od-upload-results-dark.gif#only-dark)
    <figcaption markdown>Model Results Upload</figcaption>
    </figure>

    It's also important to note down the evaluation configuration of the model being uploaded, for example:
    `iou_threshold: 0.5` or `iou_threshold: 0.3`.

    <div class="grid" markdown>
    <div markdown>
    <figure markdown>
        ![Example of a Model Configuration](../assets/images/task-od-model-config-dark.png#only-dark)
        ![Example of a Model Configuration](../assets/images/task-od-model-config-light.png#only-light)
        <figcaption>Example of a Model Configuration</figcaption>
    </figure>
    </div>
    <figure markdown>
    ![Setting Evaluation Configurations](../assets/images/task-od-evaluation-config-dark.gif#only-dark)
    ![Setting Evaluation Configurations](../assets/images/task-od-evaluation-config-light.gif#only-light)
    <figcaption>Setting Evaluation Configurations</figcaption>
    </figure>
    </div>

    You can repeat the above steps with all the other model files availible.

=== "SDK"

    The example code contains a script [`object_detection_2d/upload_results.py`](https://github.com/kolenaIO/kolena/blob/trunk/examples/dataset/object_detection_2d/object_detection_2d/upload_results.py)
    which will process raw model results from `s3://kolena-public-examples/coco-2014-val/transportation/results/raw/`
    and upload model results using the `upload_object_detection_results` function.

    ```shell
    poetry run python3 object_detection_2d/upload_results.py yolo_x
    poetry run python3 object_detection_2d/upload_results.py faster_rcnn
    ```

    Results for two models named `yolo_x` and `faster_rcnn` will appear after the upload is complete.

### Step 4: Define Object Detection Quality Standards

Once your dataset and model results are uploaded, comparing models across different scenarios and metrics becomes
very easy. Simply set up your [Quality Standards](../dataset/core-concepts/index.md#quality-standard) by
[defining test cases](../dataset/quickstart.md#define-test-cases) and
[defining performance metrics](../dataset/quickstart.md#define-metrics).

<figure markdown>
  ![Setting up object detection Quality Standards](../assets/images/task-od-qs-light.gif#only-light)
  ![Setting up object detection Quality Standards](../assets/images/task-od-qs-dark.gif#only-dark)
  <figcaption markdown>Setting up object detection Quality Standards</figcaption>
</figure>

Rather than uploading raw inferences with metrics yourself, if you upload the
[TP / FP / FN](../metrics/tp-fp-fn-tn.md) results from your own [bounding box matcher](../metrics/geometry-matching.md)
as lists of bounding boxes, Kolena makes it very easy to incorporate relevant object detection metrics.

<div class="grid" markdown>
<div style="width: 75%" markdown>
<figure markdown>
  ![OD_metrics.png](../assets/images/task-od-qs-light.png#only-light)
  ![OD_metrics.png](../assets/images/task-od-qs-dark.png#only-dark)
  <figcaption markdown>Configurable metrics for custom
  [Quality Standards](../dataset/core-concepts/index.md#quality-standard)
  </figcaption>
</figure>
</div>
<div markdown>
!!! info "Built-In Object Detection Metrics"

    Furthermore, Kolena can automatically compute common object detection metrics such as
    [precision](../metrics/precision.md) and [recall](../metrics/recall.md), making it easy to visualize an
    assortment of plots on the fly.

    ![OD_metrics.png](../assets/images/task-od-debugger-light.png#only-light)
    ![OD_metrics.png](../assets/images/task-od-debugger-dark.png#only-dark)

    In the debugger, you are able to see plots by class or by test case as shown above. For more details on
    automatic metrics and plots, please refer to documentation for
    [formatting results](../dataset/advanced-usage/formatting-your-datasets.md#formatting-results-for-object-detection).
</div>
</div>

In conclusion, evaluating object detection models is a multifaceted process based on the comparison of ground truths
and model inferences. Depending on your own project needs, custom metrics and plots should be considered.
Understanding and effectively applying evaluation metrics are essential for optimizing object detection
models to meet the demands of real-world applications.