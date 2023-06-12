---
icon: kolena/flag-16
---

# :kolena-flag-20: Core Concepts

## Workflow

Testing in Kolena is broken down by the type of ML problem you're solving, called a **workflow**. Any ML problem that
can be tested can be modeled as a workflow in Kolena. Examples include:

- [Keypoint Detection](https://github.com/kolenaIO/kolena/tree/trunk/examples/keypoint_detection) using images
- [Text Summarization](https://github.com/kolenaIO/kolena/tree/trunk/examples/text_summarization) using articles/documents
- [Age Estimation](https://github.com/kolenaIO/kolena/tree/trunk/examples/age_estimation) (regression) using images
- [Video Retrieval](https://paperswithcode.com/task/video-retrieval) using text queries on a corpus of videos

Kolena provides client bindings, [`kolena.workflow`](/reference/workflow), to define and test any arbitrary ML problem.


## Test Sample

In Kolena, "test sample" is the general term for the input to a model.

For standard computer vision (CV) models, the test sample is often a single [image][kolena.workflow.Image]. Video-based
computer vision models would have a [video][kolena.workflow.Video] test sample type, and stereo vision models would use
[image pairs][kolena.workflow.Composite]. For natural language processing models, the test sample may be a
[document][kolena.workflow.Document] or [text snippet][kolena.workflow.Text].

When [building a workflow](building-a-workflow), you can [extend](/reference/workflow/test-sample) and
[compose][kolena.workflow.Composite] these base test sample types as necessary, or use the base types directly if no
customization is required.

## Test Case

A test case is a collection of

### Best Practices

??? question "How many samples should be included in a test case?"

    The multi-model Results comparison view in Kolena takes the number of test samples within a test case into account
    when highlighting improvements and regressions. Test cases with more test samples

??? question "How many negative samples should a test case include?"

    Many workflows, such as object detection or binary classification, have a concept of "negative" samples. In object
    detection, a "negative sample" is a sample (i.e. image) that does not include any objects to be detected.

    Negative samples can have a large impact on certain metrics. To continue with the object detection example, the
    **precision** metric depends on the number of false positive detections:

    $$
    \text{Precision} := \dfrac{\text{# True Positives}}{\text{# True Positives} + \text{# False Positives}}
    $$

    Therefore, since each negative sample has some likelihood of yielding false positive detections but no likelihood of
    yielding true positive detections, negative samples will skew this precision metric downwards.

    As a general rule of thumb, we recommend including **an even balance of positive and negative samples in each test
    case.** This composition minimizes the likelihood of different metrics being heavily skewed in one direction or
    another.

## :kolena-test-suite-20: Test Suite

## :kolena-model-20: Model
