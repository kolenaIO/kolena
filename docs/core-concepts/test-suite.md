---
icon: kolena/test-suite-16
---

# :kolena-test-suite-20: Test Case & Test Suite

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
