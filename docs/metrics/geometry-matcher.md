# Geometry Matcher

Geometry matcher finds the best possible match, given the sets of the ground truth and prediction polygons for each image. The geometry matcher utility is a building block for any object detection metrics, such as precision, recall, and average precision. In this guide, we will focus on 2D object detection workflow, specifically 2D bounding box matching.

The criteria for a valid match is as follows:

1. The prediction `label` matches the ground truth `label`.
2. The prediction polygon has more than a certain % of overlap (e.g., [intersection over union (IoU)](./iou.md), [generalized IoU](https://giou.stanford.edu/GIoU.pdf)) in the area with the ground truth polygon.

Here is the pseudocode of the matching logic:

```python
for each image:
	for each label:
		use ground truths and predictions of this label
		sort predictions by descending confidence score
		for each prediction:
			find an unmatched ground truth with max IoU and IoU >= threshold
```

## Examples: 2D Bounding Boxes Matching

Let's apply the pseudocode above to the following examples of 2D object detection sample. Bounding boxes in the images below are using the following colors based on their type and the matching result:

![example legends](../assets/images/metrics-geometry-matcher-legends.png)

### Example 1

Here is an example of two ground truth and two prediction bounding boxes with a same `label` where $(\text{A}, \text{a})$ pair has a great overlap, $\text{IoU} = 0.9$, and the other pair $(\text{B}, \text{b})$ has a poor overlap, $\text{IoU} = 0.13$. Let's find out what the matched results look like in this example given that $\text{threshold}_\text{IoU} = 0.5$.

![example 1](../assets/images/metrics-geometry-matcher-example1.png)

Because prediction $\text{a}$ has a higher confidence score than prediction $\text{b}$, it gets matched first. It is pretty clear that ground truth $\text{A}$ scores the maximum IoU with prediction $\text{a}$, **and** IoU is greater than $\text{threshold}_\text{IoU}$, so **$\text{a}$ is matched with $\text{A}$**.

Next, prediction $\text{b}$ gets compared against all ground truth bounding boxes. Once again, it is clear that ground truth $\text{B}$ scores the maximum IoU with prediction $\text{b}$ but this time IoU is less than $\text{threshold}_\text{IoU}$, so **$\text{b}$ becomes an unmatched prediction**.

Now that we have checked all predictions, any ground truth bounding boxes that are not matched yet are marked as unmatched. In this case, **ground truth $\text{B}$ is the only unmatched ground truth**.

<center>

| Bounding Box(es) | Match / Unmatch |
| --- | --- |
| ($\text{A}$, $\text{a}$) | Matched Pair |
| $\text{B}$ | Unmatched GT |
| $\text{b}$ | Unmatched Prediction |

</center>

### Example 2

Let's take a look at another example with multiple classes (`Apple` and `Banana`) this time.

![example 2](../assets/images/metrics-geometry-matcher-example2.png)

Each class is evaluated independently. Starting with `Apple`, there is one ground truth ($\text{A}$) and one prediction ($\text{a}$), but these two do not overlap at all, $\text{IoU} = 0$. Because IoU is less than $\text{threshold}_\text{IoU}$, **there is no match for class `Apple`**.

For class `Banana`, there is only one prediction and no ground truth.  Therefore, **there is also no match for class `Banana`**.

<center>

| Bounding Box(es) | Match / Unmatch |
| --- | --- |
| $\text{A}$ | Unmatched GT |
| $\text{a}$ | Unmatched Prediction |
| $\text{b}$ | Unmatched Prediction |

</center>

### Example 3

Here is another example with multiple predictions overlapping with a same ground truth.

![example 3](../assets/images/metrics-geometry-matcher-example3.png)

Among the two predictions $\text{a}$ and $\text{b}$, $\text{b}$ has a higher confidence score, so $\text{b}$ gets to be matched first. IoU between ground truth $\text{A}$ and $\text{b}$ is greater than $\text{threshold}_\text{IoU}$, so they become a match.

Prediction $\text{a}$ is compared with ground truth $\text{A}$, but even though IoU is greater than $\text{threshold}_\text{IoU}$, they cannot become a match because $\text{A}$ is already matched with $\text{b}$, so prediction $\text{a}$ remains unmatched.

<center>

| Bounding Box(es) | Match / Unmatch |
| --- | --- |
| $(\text{A}, \text{b})$ | Matched Pair |
| $\text{a}$ | Unmatched Prediction |

</center>

### Example 4

Let's consider another scenario where there are multiple ground truths overlapping with a same prediction.

![example 4](../assets/images/metrics-geometry-matcher-example4.png)

Prediction $\text{a}$ has a higher IoU with ground truth $\text{B}$, so $\text{a}$ and $\text{B}$ become matched.

<center>

| Bounding Box(es) | Match / Unmatch |
| --- | --- |
| $(\text{B}, \text{a})$ | Matched Pair |
| $\text{A}$ | Unmatched GT |

</center>

## Comparison of Different Matching Logic From Popular Benchmarks
For object detection workflow, matching predictions to ground truths is a foundation step to compute detection metrics such as precision, recall, and average precision. The matching logic we have covered above is a base for the logic used in popular object detection benchmarks. In this section, we will compare different matching logic used in some of these popular benchmarks:

- [**Pascal VOC Challenge**](#pascal-voc-challenge)
- [**COCO Detection Challenge**](#coco-detection-challenge)
- [**Google Open Image V7 Competition**](#open-images-detection-challenge)
- [**ImageNet Object Localization Challenge**](#imagenet-detection-challenge)

### Pascal VOC Challenge


 that are used in popular object detection benchmarks.

The [Pascal VOC Challenge Evaluation](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/devkit_doc_08-May-2010.pdf) is considered to be the standard implementation of the geometry matcher metric. Aside from this approach, others have been used in popular object detection benchmarks, such as [COCO Detection Challenge Evaluation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py#L235), [Google Open Images V7 Competition Evaluation](https://storage.googleapis.com/openimages/web/evaluation.html), and [ImageNet Object Localization Challenge Evaluation](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/overview/evaluation).


**Difficult Flag**

The Pascal VOC benchmark dataset annotation includes a `difficult` boolean flag for each ground truth label, so that if an object is difficult to recognize from an image, this flag will be set. Any ground truth with the `difficult` flag and any prediction that is matched with a `difficult` ground truth will be skipped during the matching operation. In other words, these ground truths or predictions that are matched with them will not be included the matched results.

The pseudocode of Pascal VOC benchmark’s bounding box matcher would be as follows:

```python
for each image:
	for each label:
		sort predictions by descending confidence score
		for each prediction:
			find an unmatched ground truth with max IoU and IoU >= threshold
			if matched with a difficult ground truth
				continue
```

## Other Implementations

While Pascal VOC’s matcher is considered to be the standard implementation, there are many other popular object detection benchmarks with some slight **variations** applied to the standard implementation.  In this section, we will look at how the three other major benchmarks (**COCO Detection**, Google’s **Open Images**, and **ImageNet**) vary from the standard implementation.

### COCO Detection Challenge

COCO evaluation has a couple of more things to consider when matching prediction geometries. First, the way to handle `difficult` object detection is slightly different in COCO evaluation. Similarly to how `difficult` ground truths are treated in the standard implementation, COCO labels its ground truths with `iscrowd` metadata to specify when a ground truth includes multiple objects, and these ground truths and any predictions that matched with them are excluded from the matched results. It is to avoid penalizing models for failing to detect objects in a crowded scene. A small detailed difference here between the standard implementation and COCO evaluation is that these `iscrowd` ground truths can be matched multiple times with different predictions whereas `difficult` ground truths from Pascal VOC can only be matched once.

Another major difference between COCO and Pascal VOC matching algorithms is how a range of `area`  for both ground truth and prediction is used in evaluation. COCO evaluates different detector models across different sizes of objects (e.g., `small`, `medium`, and `large` objects) by ignoring any objects (either ground truth or prediction) outside of the specified range of `area`.  Ground truth geometries that are outside of the set range are filtered out prior to the matching, and any unmatched predictions outside of the set range are ignored.

The pseudocode for COCO matching algorithm is as follows:

```python
for area_range in [small, medium, large]:
	for each image:
		for each label:
			sort predictions by descending confidence score
			sort ground truth that ground truth outside of area_range or iscrowd come last
			for each prediction:
				find an unmatched or (already matched but iscrowd) ground truth with max IoU and IoU >= threshold
				if matched with a ground truth outside of area_range or iscrowd
					set the matched pair as ignore
			set any unmatched predictions outside of area_range as ignore
```

### Open Images Detection Challenge

The Open Images Challenge uses a variant of the standard implementation, but there are two key differences in the matching algorithm. The first is with the way that the images are annotated in this dataset. They’re annotated with **positive** **image-level** labels, indicating certain object classes are preset, and with **negative** **image-level** labels, indicating certain classes are absent. Therefore, for fair evaluation, all unannotated classes are excluded from evaluation in that image, so if a prediction has a class label that is unannotated on that image, it is ignored:

![Handling non-exhaustive image-level labeling from [Open Image Challenge Evaluation](https://storage.googleapis.com/openimages/web/evaluation.html) [[4]](https://www.notion.so/Utility-Geometry-Matcher-67844e62f3e6441f87573d0ddae1c1bc?pvs=21)](Utility%20Geometry%20Matcher%2067844e62f3e6441f87573d0ddae1c1bc/Untitled%205.png)

Handling non-exhaustive image-level labeling from [Open Image Challenge Evaluation](https://storage.googleapis.com/openimages/web/evaluation.html) [[4]](https://www.notion.so/Utility-Geometry-Matcher-67844e62f3e6441f87573d0ddae1c1bc?pvs=21)

The second difference is with handling `group-of` boxes, which is similar to `iscrowd` annotation from COCO but is not just simply ignored. If at least one prediction is inside the `group-of` box, then it is considered to be a match. Otherwise, the `group-of` box is considered as a false negative (FN). Multiple correct predictions inside the same `group-of` box still count as a single match:

![Handling group-of boxes from [Open Image Challenge Evaluation](https://storage.googleapis.com/openimages/web/evaluation.html) [[4]](https://www.notion.so/Utility-Geometry-Matcher-67844e62f3e6441f87573d0ddae1c1bc?pvs=21)](Utility%20Geometry%20Matcher%2067844e62f3e6441f87573d0ddae1c1bc/Untitled%206.png)

Handling group-of boxes from [Open Image Challenge Evaluation](https://storage.googleapis.com/openimages/web/evaluation.html) [[4]](https://www.notion.so/Utility-Geometry-Matcher-67844e62f3e6441f87573d0ddae1c1bc?pvs=21)

The pseudocode for the open images matching algorithm is as follows:

```python
for each image:
	for each label:
		filter out predictions with labels that are negative image-level labels
		sort predictions by descending confidence score
		for each prediction:
			find an unmatched non-group-of ground truth with max IoU and IoU >= threshold
			if matched with difficult ground truth:
				set the matched pair as difficult
		for each unmatched prediction:
			find a group-of ground truth with max IoU and IoU >= threshold
			if matched with group-of ground truth:
				set the matched pair as group-of and take the maximum score of matched predictions on this ground truth
```

### ImageNet Detection Challenge

The ImageNet evaluation for object detection challenge is a bare-bones version of the standard implementation that checks for two things: one, that the prediction has a matching class label with the ground truth, and two, that the prediction has a more than 50% match in the area with the ground truth. The benchmark is not evaluated based on the count of true positive (TP), false positive (FP), or false negative (FN), like a traditional object detection evaluation is. Instead, it measures whether there is any “error” in an image in a binary form, `min_error`.  If there's a match in an image, then `min_error` for the image is `0`; otherwise, `min_error` is `1`.

The pseudocode for ImageNet matching algorithm is as follows:

```python
for each image:
	for each prediction:
		for each ground truth:
			if label matches and IoU > 0.5:
				error = 0
			else
				error = 1
		find a ground truth with minimum error
	find a set with minimum error
```

# Limitations and Biases

The standard implementation of geometry matcher appears to have an undesirable behavior when there are many overlapping ground truths and predictions with high confidence scores:

**Example of greedy matching**

![Untitled](Utility%20Geometry%20Matcher%2067844e62f3e6441f87573d0ddae1c1bc/Untitled%207.png)

When there are two ground truths and two predictions, one prediction (`b`) with a higher score overlaps well with both ground truths (`A` and `B`), and the other one (`a`) with a lower score overlaps well with just one ground truth (`B`). Because `IoU(B, b)` is greater than `IoU(A, b)`, prediction `b` is matched with ground truth `B`, so prediction `a` will fail to be matched. This greedy matching behavior results in a higher FP count in this type of scenario. Ideally, prediction `a` matches with ground truth `B` , and prediction `b` matches with ground truth `A`, resulting in no FPs. The standard implementation optimizes for the higher confidence `score` and maximum `IoU`.

Another behavior to note here is that it is possible to get different matching results depending on the ground truth order when there are multiple ground truths overlapping with a prediction with the same `IoU` or depending on the prediction order when there are multiple predictions overlapping with a ground truth with the same `score`.

**Example of matching dependent on ground truth order**

![Untitled](Utility%20Geometry%20Matcher%2067844e62f3e6441f87573d0ddae1c1bc/Untitled%208.png)

The image above is similar to the last example, but this time they all share the same `IoU` and `score`. If the ground truths are listed as `[A, B]` and the predictions as `[a, b]`, prediction `a` is matched with `B`, so prediction `b` is matched with `A`. If prediction order changes to `[b, a]`, now prediction `a` may or may not be matched, it completely depends on ground truth order. If `A` comes before `B`, prediction `b` is matched with `A` and `a` can be matched with `B`. However, if `B` comes before `A`, prediction `b` is matched with `B` instead, and then `a` is left with no match.

# References

[1] [Pascal VOC Challenge](http://host.robots.ox.ac.uk/pascal/VOC/)

[2] [Pascal VOC Challenge Dev Kit Doc](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/devkit_doc_08-May-2010.pdf)

[3] [COCO Detection Challenge](https://cocodataset.org)

[4] [Open Images Dataset Detection Evaluation](https://storage.googleapis.com/openimages/web/evaluation.html)

[5] [ImageNet Object Localization Challenge Evaluation](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/overview/evaluation)

# Kolena API

[`kolena.workflow.metrics.match_inferences`](https://docs.kolena.io/reference/workflow/metrics/#kolena.workflow.metrics.match_inferences)

[`kolena.workflow.metrics.match_inferences_multiclass`](https://docs.kolena.io/reference/workflow/metrics/#kolena.workflow.metrics.match_inferences_multiclass)
