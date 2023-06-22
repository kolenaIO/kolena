---
search:
  exclude: true
---

# Geometry Matcher

Geometry matcher finds the best possible match, given the sets of the ground truth and prediction polygons for each
image. The geometry matcher utility is a building block for any object detection metrics, such as precision, recall,
and average precision. In this guide, we will focus on 2D object detection workflow, specifically 2D bounding box
matching.

The criteria for a valid match is as follows:

1. The prediction label matches the ground truth label.
2. The prediction polygon has more than or equal to a certain overlap (i.e. IoU threshold) with the ground truth polygon.

Here is the general matching logic:

1. Loop through all images in your dataset;
2. Loop through all labels;
3. Get predictions and ground truths with the evaluating label;
4. Sort predictions by descending confidence score;
5. Check against all ground truths and find a ground truth that results in maximum IoU;
6. Check for the following criteria for a valid match:
	1. This ground truth is not matched yet AND
	2. The IoU is greater than or equal to the IoU threshold;
7. Repeat 5-6 on the next prediction;


## Examples: 2D Bounding Boxes Matching

Let's apply the logic above to the following examples of 2D object detection sample. Bounding boxes in the images below
are using the following colors based on their type and the matching result:

![example legends](../assets/images/metrics-matcher-legends.png)

### Example 1

Here is an example of two ground truth and two prediction bounding boxes with a same label where
(A, a) pair has a great overlap, IoU of 0.9, and the other pair (B, b) has a
poor overlap, IoU of 0.13. Let's find out what the matched results look like in this example with a
IoU threshold of 0.5.

![example 1](../assets/images/metrics-matcher-example1.png)

Because prediction a has a higher confidence score than prediction b, it gets matched first. It is
pretty clear that ground truth A scores the maximum IoU with prediction a, and IoU is greater than
IoU threshold, so a is **matched** with A.

Next, prediction b gets compared against all ground truth bounding boxes. Once again, it is clear that ground
truth B scores the maximum IoU with prediction b but this time IoU is less than the IoU threshold,
so b becomes an **unmatched prediction**.

Now that we have checked all predictions, any ground truth bounding boxes that are not matched yet are marked as
unmatched. In this case, ground truth B is the only **unmatched ground truth**.

<center>

| Bounding Box(es) | Match / Unmatch |
| --- | --- |
| (A, a) | Matched Pair |
| B | Unmatched GT |
| b | Unmatched Prediction |

</center>

### Example 2

Let's take a look at another example with multiple classes (`Apple` and `Banana`) this time.

![example 2](../assets/images/metrics-matcher-example2.png)

Each class is evaluated independently. Starting with `Apple`, there is one ground truth A and one prediction
a, but these two do not overlap at all, IoU of 0.0. Because IoU is less than the IoU threshold, there is **no match**
for class `Apple`.

For class `Banana`, there is only one prediction and no ground truth. Therefore, there is also **no match** for class
`Banana`.

<center>

| Bounding Box(es) | Match / Unmatch |
| --- | --- |
| A | Unmatched GT |
| a | Unmatched Prediction |
| b | Unmatched Prediction |

</center>

### Example 3

Here is another example with multiple predictions overlapping with a same ground truth.

![example 3](../assets/images/metrics-matcher-example3.png)

Among the two predictions a and b, b has a higher confidence score, so b gets to be matched first. IoU between
ground truth A and b is greater than the IoU threshold, so they become a **match**.

Prediction a is compared with ground truth A, but even though IoU is greater than the IoU threshold, they cannot become
a match because A is already matched with b, so prediction a remains **unmatched**.

<center>

| Bounding Box(es) | Match / Unmatch |
| --- | --- |
| (A, b) | Matched Pair |
| a | Unmatched Prediction |

</center>

### Example 4

Let's consider another scenario where there are multiple ground truths overlapping with a same prediction.

![example 4](../assets/images/metrics-matcher-example4.png)

Prediction a has a higher IoU with ground truth B, so a and B become matched.

<center>

| Bounding Box(es) | Match / Unmatch |
| --- | --- |
| (B, a) | Matched Pair |
| A | Unmatched GT |

</center>

## Comparison of Different Matching Logic From Popular Benchmarks

For object detection workflow, matching predictions to ground truths is a foundation to compute detection metrics
such as precision, recall, and average precision. The matching logic we have covered above is a base for popular object
detection benchmarks' evaluation. In this section, we will compare different matching logics from some of these
popular benchmarks:

- [**Pascal VOC Challenge**](#pascal-voc-challenge)
- [**COCO Detection Challenge**](#coco-detection-challenge)
- [**Google Open Image V7 Competition**](#open-images-detection-challenge)

### Pascal VOC Challenge

The [Pascal VOC benchmark](http://host.robots.ox.ac.uk/pascal/VOC/) dataset includes a `difficult` boolean
annotation for each ground truth, used to differentiate objects that are difficult to recognize from an image.
Any ground truth with the `difficult` flag and any predictions that are matched with a `difficult` ground truth will
be ignored in the matching process. In other words, these ground truths or predictions that are matched with them are
**excluded** in the matched results. Hence, models will not be penalized for failing to detect these `difficult`
objects.

Another difference that is noteworthy is how Pascal VOC outlines the IoU criteria for a valid match. According to the
evaluation section (4.4) in
[development kit doc](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/devkit_doc_08-May-2010.pdf), IoU must **exceed**
the IoU threshold to be considered as a valid match.

??? example "Pascal VOC's Matching Logic"

	1. Loop through all images in your dataset;
	2. Loop through all labels;
	3. Get predictions and ground truths with the evaluating label;
	4. Sort predictions by descending confidence score;
	5. Check against all ground truths and find a ground truth that results in maximum IoU;
	6. Check for the following criteria for a valid match:
		1. This ground truth is not matched yet AND
		2. The IoU is **greater than** the IoU threshold;
	7. **If matched with a `difficult` ground truth, ignore**;
	8. Repeat 5-7 on the next prediction;


### COCO Detection Challenge

[COCO detection challenge](https://cocodataset.org) evaluation has couple more things to consider when matching
prediction geometries. First, similarly to how `difficult` ground truths are treated in Pascal VOC, COCO benchmark
labels its ground truths with `iscrowd` annotation to specify when a ground truth includes multiple objects, and these
ground truths and any predictions that matched with them are **excluded** from the matched results. It is to avoid
penalizing models for failing to detect objects in a crowded scene.

??? example "COCO's Matching Logic"

	1. Loop through all images in your dataset;
	2. Loop through all labels;
	3. Get predictions and ground truths with the evaluating label;
	4. Sort predictions by descending confidence score;
	5. Check against all ground truths and find a ground truth that results in maximum IoU;
	6. Check for the following criteria for a valid match:
		1. This ground truth is not matched yet AND
		2. The IoU is greater than or equal to the IoU threshold;
	7. **If matched with a `iscrowd` ground truth, ignore**;
	8. Repeat 5-7 on the next prediction;


### Open Images Detection Challenge

The [Open Images V7 Challenge](https://storage.googleapis.com/openimages/web/evaluation.html) evaluation introduces two
key differences in the matching logic.

The first is with the way that the images are annotated in this dataset. They
are annotated with **positive** **image-level** labels, indicating certain object classes are preset, and with
**negative** **image-level** labels, indicating certain classes are absent. Therefore, for fair evaluation, all
unannotated classes are **excluded** from evaluation in that image, so if a prediction has a class label that is
unannotated on that image, this prediction is excluded in the matching results.

![An example of non-exhaustive labeling](../assets/images/metrics-matcher-oid-non-exhaustive.jpg)

<p style="text-align: center; color: gray;">
    An example of non-exhaustive image-level labeling from
	<a href="https://storage.googleapis.com/openimages/web/evaluation.html">Open Image Challenge Evaluation</a>
</p>

The second difference is with handling `group-of` boxes, which is similar to `iscrowd` annotation from
[COCO](#coco-detection-challenge) but is not just simply ignored. If at least one prediction is inside the `group-of`
box, then it is considered to be a match. Otherwise, the `group-of` box is considered as an unmatched ground truth.
Also, multiple correct predictions inside the same `group-of` box still count as a single match:

![An example of group-of boxes](../assets/images/metrics-matcher-oid-group-of.jpg)

<p style="text-align: center; color: gray;">
	An example of group-of boxes from
	<a href="https://storage.googleapis.com/openimages/web/evaluation.html">Open Image Challenge Evaluation</a>
</p>

??? example "Open Image's Matching Logic"

	1. Loop through all images in your dataset;
	2. Loop through all **positive image-level** labels;
	3. Get predictions and ground truths with the evaluating label;
	4. Sort predictions by descending confidence score;
	5. Check against all **non-`ground-of`** ground truths and find a ground truth that results in maximum IoU;
	6. Check for the following criteria for a valid match:
		1. This ground truth is not matched yet AND
		2. The IoU is greater than or equal to the IoU threshold;
	7. **If matched with a `difficult` ground truth, ignore**;
	8. Repeat 5-7 on the next prediction;
	9. **Loop through all unmatched predictions;**
	10. **Check against all `group-of` ground truths and find a ground truth that results in maximum IoU;**
	11. **Check for the matching criteria (6);**
	12. **Repeat 10-11 on the next unmatched prediction;**


## Limitations and Biases

The standard matching logic appears to have an undesirable behavior when there are many overlapping ground truths and
predictions with high confidence scores due to its **greedy matching**. Because the logic optimizes for higher confidence
score and maximum IoU, it can potentially miss valid matches by matching a nonoptimal pair, resulting in a poorer
matching performance.

??? example "Example: Greedy Matching"

	![An example of greedy matching](../assets/images/metrics-matcher-greedy-matching.png)

	When there are two ground truths and two predictions, one prediction b with a higher score overlaps well with
	both ground truths A and B, and the other one, a, with a lower score overlaps well with just one ground truth B.
	Because IoU of B and b is greater than IoU of A and b, prediction b is matched with ground truth B, causing
	prediction a to fail to match. This greedy matching behavior results in a higher false positive count in this type
	of scenario. Ideally, prediction a matches with ground truth B, and prediction b matches with ground truth A,
	resulting in no FPs.

Another behavior to note here is that it is possible to get different matching results depending on the **ground truth**
**order** when there are multiple ground truths overlapping with a prediction with the equal IoU or depending on the
**prediction order** when there are multiple predictions overlapping with a ground truth with the equal confidence score.

??? example "Example: Different Matching Results with GT Order Change"
	![An example of GT ordering](../assets/images/metrics-matcher-gt-order.png)

	The three pairs of ground truth and prediction have **same IoU** and both predictions have **same confidence score**.
	If the ground truths are ordered as `[A, B]` and the predictions as `[a, b]`, prediction a is matched with B first,
	so prediction b gets matched with A. If the prediction order changes to `[b, a]`, now prediction a may or may not be
	matched with any ground truth. The matched result can change depending on the ground truth order. If A is evaluated
	before B, prediction b is matched with A, and a can be matched with B. However, if B comes before A, prediction b
	is matched with B instead, leaving prediction a with no match.


## Kolena API

[`kolena.workflow.metrics.match_inferences`](https://docs.kolena.io/reference/workflow/metrics/#kolena.workflow.metrics.match_inferences)

[`kolena.workflow.metrics.match_inferences_multiclass`](https://docs.kolena.io/reference/workflow/metrics/#kolena.workflow.metrics.match_inferences_multiclass)
