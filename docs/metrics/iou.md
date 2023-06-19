# Intersection over Union (IoU)

Intersection over Union (IoU) measures the ratio of the intersection and the union between two instances, ranging from 0 to 1 where 1 indicates a perfect match.  It is one of the metrics that measure the similarity between two instances, often used to compare predictions to ground truths.

As the name suggests, the IoU of two instances ($A$ and $B$) is defined as:

$$IoU(A, B) = \frac {A \cap B} {A \cup B}$$


## When Do I Use IoU?
It is often used to compare two geometries (e.g., bounding boxes, polygons) or segmentation masks in object detection, instance segmentation, or semantic segmentation models. In multi-label classification, IoU, more likely known as the [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index), is used to compare set of prediction labels for a sample to the corresponding set of ground truth labels. Moreover, there are workflows such as [action detection](https://paperswithcode.com/task/action-detection) and [video moment retrieval](https://paperswithcode.com/task/moment-retrieval) where IoU measures the **temporal** overlap between two time-series snippets.


Because IoU can be used on various types of data, let's look at how the metric is defined for some of these data types:

- [**2D Axis-Aligned Bounding Boxes**](#2d-axis-aligned-bounding-box)
- [**Segmentation Mask**](#segmentation-mask)
- [**Set of Labels**](#set-of-labels)


### 2D Axis-Aligned Bounding Box

Let's consider two 2D axis-aligned bounding boxes, $\text{A}$ and $\text{B}$, and notice the highlighted overlap region, which is always going to be a 2D axis-aligned bounding box.

![An example of a 2D axis-aligned bounding box](../assets/images/metrics-iou-2dbbox.png)


In order to compute IoU for two 2D axis-aligned bounding boxes, the first step is identifying the area of the intersection box, $(\text{A} \cap \text{B})$. The two coordinates of the intersection box, top-left and bottom-right corners, can be defined as:

$$
\text{A} \cap \text{B}\,_{\text{topleft}} = ( max \left( x_{a1}, \, x_{b1} \right), \, max \left( y_{a1}, \, y_{b1} \right))
$$

$$
\text{A} \cap \text{B}\,_{\text{bottomright}} = (min \left( x_{a2}, \, x_{b2} \right), \, min \left(y_{a2}, \, y_{b2} \right))
$$

Once the intersection box $(\text{A} \cap \text{B})$ is identified, the area of the union, $(\text{A} \cup \text{B})$, is simply a sum of the area of $\text{A}$ and ${\text{B}}$ minus the area of the intersection box. Finally, IoU is calculated by taking the ratio of the area of intersection box and the area of the union region.

$$
\text{area} \left( \text{A} \cup \text{B} \right) = \text{area} \left( \text{A} \right) + \text{area} \left( \text{B} \right) - \text{area} \left( \text{A} \cap \text{B} \right)
$$

#### Examples: IoU of 2D Bounding Boxes
The following examples show what IoU values look like in different scenarios with 2D bounding boxes:

**Example 1: overlapping bounding boxes**

![An example of overlapping bounding boxes](../assets/images/metrics-iou-example1.png)

$$
\begin{align}
\text{IoU} \left( \text{A}, \, \text{B} \right)
&= \frac {(10 - 5) \times (10 - 5)} {10 \times 10 + (15 - 5) \times (15 - 5) - (10 - 5) \times (10 - 5)} \\[1em]
&= \frac {25} {100 + 100 - 25} \\[1em]
&= 0.143
\end{align}
$$

**Example 2: non-overlapping bounding boxes**

![An example of non-overlapping bounding boxes](../assets/images/metrics-iou-example2.png)

$$
\begin{align}
\text{IoU} \left( \text{A}, \, \text{B} \right)
&= \frac {0} {10 \times 10 + (15 - 10) \times (15 - 10) - 0} \\[1em]
&= \frac {0} {100 + 25 - 0} \\[1em]
&= 0.0
\end{align}
$$


**Example 3: completely matching bounding boxes**

![An example of completely matching bounding boxes](../assets/images/metrics-iou-example3.png)

$$
\begin{align}
\text{IoU} \left( \text{A}, \, \text{B} \right)
&= \frac {10 \times 10} {10 \times 10 + 10 \times 10 - 10 \times 10} \\[1em]
&= \frac {100} {100 + 100 - 100} \\[1em]
&= 1.0
\end{align}
$$


### Segmentation Mask

**A segmentation mask** is a 2D image where each pixel is a class label commonly used in semantic segmentation tasks. The prediction shape matches the ground truth shape (width and height), with a channel depth equivalent to the number of class labels to be predicted. Each channel is a binary mask that labels areas where a specific class is present:

![An example of segmentation mask](../assets/images/metrics-iou-seg-mask.png)
<p style="text-align: center; color: gray;">
    From left to right: the original RGB image, the ground truth segmentation mask, and the prediction segmentation mask
</p>


The IoU metric measures the **intersection** (the number of pixels **common** between the ground truth and prediction masks, **true positive (TP)**) divided by the **union** (the **total** number of pixels present across **both** masks, **TP + false negative (FN) + false positive (FP)**). And here is the formula for the IoU metric for a segmentation mask:

$$
\text{IoU} \left( \text{A}, \, \text{B} \right) = \frac {\text{TP}} {\text{TP} + \text{FN} + \text{FP}}
$$

Let’s look at what TP, FP, and FN look like on a segmentation mask:

![An example of segmentation mask with results](../assets/images/metrics-iou-seg-mask-results.png)
<p style="text-align: center; color: gray;">
    From left to right: the ground truth segmentation mask, the prediction segmentation mask, and the overlay with TP, FP, and FN labeled
</p>

From the cat image shown above, when you overlay the ground truth and prediction masks, the pixels that belong to both masks are TP. The pixels that only exist in the ground truth mask are FNs, and the pixels that only exist in the prediction mask are FPs. Let's consider the following pixel counts for each category:

| # True Positives | # False Positives | # False Negatives |
| --- | --- | --- |
| 100 | 25 | 50 |

Then the IoU becomes:

$$
\begin{align}
\text{IoU} &= \frac {100} {(100 + 25 + 50)} \\[1em]
&= 0.571
\end{align}
$$

### Set of Labels

The **set of labels** used in multi-label classification tasks is often a [binarized](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.label_binarize.html) list with a number of label elements, for example, let’s say there are three classes, `Airplane`, `Boat`, and `Car`, and a sample is labeled as `Boat` and `Car`. The binary set of labels would then be $[0, 1, 1]$, where each element represents each class, respectively.

Similar to the segmentation mask, the IoU or Jaccard index metric for the ground truth/prediction labels would be the size of the **intersection** of the two sets (the number of labels **common** between two sets, or TP) divided by the size of the **union** of the two sets (the total number of labels present in both sets, **TP + FN + FP**):

$$
IoU(A, B) = \frac {TP} {TP + FN + FP}
$$

The IoU for multi-label classification is defined per class. This technique, also known as one-vs-the-rest (OVR), evaluates each class as a binary classification problem. Per-class IoU values can then be aggregated using different [averaging methods](./averaging-methods.md). The popular choice for this workflow is **macro**, so let’s take a look at examples of different averaged IoU/Jaccard index metrics for multi-class multi-label classification:

**Example: macro IoU of ground truth and prediction sets of labels: `Airplane`, `Boat`, `Car`**

$$
\text{A} = [[1, 1, 1], \, [1, 0, 0], \, [0, 1, 1]]
$$

$$
\text{B} = [[0, 1, 0], \, [1, 0, 0], \, [1, 1, 0]]
$$

$$
\begin{align}
\text{IoU}_\text{macro} &= \frac {\text{IoU}_\texttt{Airplane} + \text{IoU}_\texttt{Boat} + \text{IoU}_\texttt{Car}} {3} \\[1em]
&= \frac {\frac 1 3 + \frac 2 2 + \frac 0 2} {3} \\[1em]
&= 0.444
\end{align}
$$


## Limitations and Biases

IoU works great to measure the overlap between two sets, whether they are types of geometry or a list of labels. However, this metric cannot be directly used to measure the overlap of a prediction and `iscrowd` ground truth, which is an annotation used to label a large groups of objects (e.g., a crowd of people). Therefore, the prediction is expected to take up a small portion of the ground truth region, resulting in a low IoU score and a pair not being a valid match. In this scenario, a variation of IoU, called [intersection over foreground (IoF)](https://github.com/open-mmlab/mmdetection/issues/393), is preferred. This variation is used when there are ground truth regions you want to ignore in evaluation, such as `iscrowd`.

The second limitation of IoU is measuring the localization performance of non-overlaps. IoU ranges from 0 (no overlap) to 1 (complete overlap), so when two bounding boxes have zero overlap, it’s hard to tell how bad the localization performance is solely based on IoU. There are variations of IoU, such as [signed IoU (SIoU)](https://arxiv.org/pdf/1905.12365.pdf) and [generalized IoU (GIoU)](https://giou.stanford.edu/GIoU.pdf), that aim to measure the localization error even when there is no overlap. These metrics can replace IoU metric if the objective is to measure the localization performance of non-overlaps.


## Kolena API

[`kolena.workflow.metrics.iou`](https://docs.kolena.io/reference/workflow/metrics/#kolena.workflow.metrics.iou)
