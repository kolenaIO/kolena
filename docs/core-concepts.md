---
icon: kolena/flag-16
---

# :kolena-flag-20: Core Concepts

## Workflow

Testing in Kolena is broken down by the type of ML problem you're solving, called a **workflow**. Any ML problem that
can be tested can be modeled as a workflow in Kolena. Examples include:

<div class="grid cards" markdown>
- [:kolena-keypoint-detection-20: Keypoint Detection](https://github.com/kolenaIO/kolena/tree/trunk/examples/keypoint_detection) using images
- [:kolena-text-summarization-20: Text Summarization](https://github.com/kolenaIO/kolena/tree/trunk/examples/text_summarization) using articles/documents
- [:kolena-age-estimation-20: Age Estimation](https://github.com/kolenaIO/kolena/tree/trunk/examples/age_estimation) (regression) using images
- [:kolena-video-20: Video Retrieval](https://paperswithcode.com/task/video-retrieval) using text queries on a corpus of videos
</div>

With the [`kolena.workflow`](/reference/workflow) client module, any arbitrary ML problem can be defined as a workflow
and tested on Kolena.


## Test Sample

In Kolena, "test sample" is the general term for the input to a model.

For standard computer vision (CV) models, the test sample is often a single [image][kolena.workflow.Image]. Video-based
computer vision models would have a [video][kolena.workflow.Video] test sample type, and stereo vision models would use
[image pairs][kolena.workflow.Composite]. For natural language processing models, the test sample may be a
[document][kolena.workflow.Document] or [text snippet][kolena.workflow.Text].

When [building a workflow](building-a-workflow), you can [extend](/reference/workflow/test-sample) and
[compose][kolena.workflow.Composite] these base test sample types as necessary, or use the base types directly if no
customization is required.

### Metadata

Any additional information associated with a test sample, e.g. details about how it was collected, can be included as
[metadata][kolena.workflow.Metadata]. We recommend uploading any and all metadata that you have available, as metadata
can be useful for searching through data in the Studio, interpreting model results, and creating new test cases.

## Test Case

A test case is a collection of [test samples](#test-sample) and their associated ground truths. Test cases can be
thought of as benchmark datasets, or smaller slices of a benchmark dataset.

### Test Case Best Practices

??? question "How many samples should be included in a test case?"

    While there's no one-size-fits-all answer, we usually recommend including at least 100 samples in each test case.
    Smaller test cases can be used to provide a very rough signal about the presence or absence of a model beahvior, but
    shouldn't be relied upon for much more than a directional indication of performance.

    The multi-model Results comparison view in Kolena takes the number of test samples within a test case into account
    when highlighting improvements and regressions. The larger the test case, the smaller the ∆ required to consider a
    change from one model to another as "significant."

??? question "How many negative samples should a test case include?"

    Many workflows, such as object detection or binary classification, have a concept of "negative" samples. In object
    detection, a "negative sample" is a sample (i.e. image) that does not include any objects to be detected.

    Negative samples can have a large impact on certain metrics. To continue with the object detection example, the
    **precision** metric depends on the number of false positive detections:

    $$
    \text{Precision} := \dfrac{\text{# True Positives}}{\text{# True Positives} + \text{# False Positives}}
    $$

    Therefore, since each negative sample has some likelihood of yielding false positive detections but no likelihood of
    yielding true positive detections, adding negative samples to a test case may decrease aggregate precision values
    computed across the test case.

    As a general rule of thumb, we recommend including **an even balance of positive and negative samples in each test
    case.** This composition minimizes the likelihood of different metrics being heavily skewed in one direction or
    another.

## :kolena-test-suite-20: Test Suite

A test suite is a collection of test cases. Models are tested on on test suites.

### Test Suite Best Practices

!!! question "How do I map my existing benchmark into a test suite?"

    a

??? question "How many test cases should a test suite include?"

    Anywhere from 1 to thousands

## :kolena-model-20: Model

A model can be thought of as a deterministic transformation from [test samples](#test-sample) to inferences.

### Model Best Practices

!!! tip "Ensure that models are deterministic"

    To preserve reproducibility, ensure that models tested in Kolena are deterministic.

    This is particularly important for generative models. If your model has a random seed parameter, consider including
    the random seed value used for testing as a piece of metadata attached to the model.
