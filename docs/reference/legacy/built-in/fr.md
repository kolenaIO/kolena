---
icon: kolena/fr-16
search:
  exclude: true
---

# :kolena-fr-20: `kolena.fr`

!!! warning "Legacy Warning"

    The `kolena.fr` module is considered **legacy** and should not be used for new projects.

    Please see `kolena.workflow` for customizable and extensible definitions to use for all new projects.

![Example Face Recognition (1:1) image pair from the Labeled Faces in the Wild dataset.](../../../assets/images/face-recognition.jpg)

Face Recognition (1:1) workflow is built to test models answering the question: _do these two images depict the same
person_? The terminology and methodology are adapted from the [NIST FRVT 1:1](https://pages.nist.gov/frvt/html/frvt11.html)
challenge.

The Face Recognition (1:1) workflow is also referred to as **Face Verification**.

## Terminology

| Term                            | Definition                                                                                                                   |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Embedding**                   | The embedding extracted by your model representing the face depicted in an image                                             |
| **Similarity**                  | The similarity score computed from two embeddings extracted by your model, where higher = more similar, lower = less similar |
| **Threshold**                   | A threshold applied to computed similarity scores, above which two faces are considered the same                             |
| **Genuine Pair**                | Two images depicting the same person                                                                                         |
| **Imposter Pair**               | Two images depicting different people                                                                                        |
| **False Match (FM)**            | An incorrect model classification of an imposter pair as a genuine pair                                                      |
| **False Match Rate (FMR)**      | The percentage of imposter pairs that are incorrectly classified as genuine pairs (i.e. similarity is above threshold)       |
| **False Non-Match (FMR)**       | An incorrect model classification of a genuine pair as an imposter pair                                                      |
| **False Non-Match Rate (FNMR)** | The percentage of genuine pairs that are incorrectly classified as imposter pairs (i.e. similarity is below threshold)       |
| **Baseline**                    | The test case(s) against which similarity score thresholds are computed ([see below](fr.md#test-suite-baseline))      |

## Test Suite Baseline

In the Face Recognition (1:1) workflow, similarity score thresholds are computed based on target false match rates over
a special section of your test suite known as the **baseline**.

In the simple standard case, you typically want the baseline to be the entire test suite. However, having control over
the baseline allows you to define test suites that answer questions like:

- How does my model perform on a certain population when the thresholds are computed using a different population (i.e.
  FMR/FNMR shift)?
- How does my model perform in different deployment conditions (e.g. low lighting, infrared) when using thresholds
  computed from standard conditions? (i.e. can my model generalize to unseen scenarios?)

## Model Pipeline

The Face Recognition (1:1) workflow is built to accommodate standard face recognition model pipelines with the following
steps:

1. Face detection (using object detection model)
2. Face alignment (using landmark detection model)
3. Feature extraction (using embedding extraction model)
4. Feature comparison (using embeddings comparison algorithm)

## Multiple Faces in a Single Image

The Face Recognition (1:1) workflow supports extraction of any number of faces from a given image during testing,
following the methodology of the [NIST FRVT 1:1 Verification](https://pages.nist.gov/frvt/html/frvt11.html)
specification section [Accuracy Consequences of Recognizing Multiple Faces in a Single Image](https://pages.nist.gov/frvt/html/slides/multiple_faces_single_image.pdf).

During testing, any number between zero (i.e. failure to enroll) and N embeddings may be uploaded for a given image.
When computing similarity scores between embeddings from two images in an image pair, all combinations of embeddings
extracted from the left and the right image in the pair are computed. When computing metrics, only the highest
similarity score between different embeddings in the image pair is considered.

## Quick Links

- [`kolena.fr.TestImages`][kolena.fr.TestImages]: register new images for testing
- [`kolena.fr.TestCase`][kolena.fr.TestCase]: create and manage test cases
- [`kolena.fr.TestSuite`][kolena.fr.TestSuite]: create and manage test suites
- [`kolena.fr.TestRun`][kolena.fr.TestRun]: test models on test suites
- [`kolena.fr.Model`][kolena.fr.Model]: create models for testing

::: kolena.fr.model
    options:
      members_order: alphabetical

::: kolena.fr.test_case
::: kolena.fr.test_images
::: kolena.fr.test_run
::: kolena.fr.test_suite

## Data Types

::: kolena.fr.datatypes
