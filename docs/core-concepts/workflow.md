---
icon: kolena/workflow-16
---

# :kolena-workflow-20: Workflows

Testing in Kolena is broken down by the type of ML problem you're solving, called a **workflow**.

---
---
---

### Supported Workflows

Kolena provides client bindings to define and test any arbitrary ML problem.

You can test any type of model on any type of data (e.g. images, video snippets, pairs of images, documents)
using the exact metrics that fit your problem by [**building a workflow**](#building-a-workflow).

#### Built-in Workflows

In addition to the general-purpose workflow builder tooling provided by the client, certain common computer vision
workflows have built-in support on Kolena:

- [**Object Detection**](#built-in-workflow-object-detection): localization and classification of objects
  using rectangular bounding boxes.
- [**Instance Segmentation**](#built-in-workflow-object-detection): localization and classification of
  objects using arbitrary polygons.
- [**Classification**](#built-in-workflow-classification): binary, multi-class, and multi-label
  classification.
- [**Face Recognition (1:1)**](#built-in-workflow-face-recognition-1-1): face recognition verification
  following the [NIST FRVT 1:1](https://pages.nist.gov/frvt/html/frvt11.html) evaluation framework.

## Building a Workflow

Any model can be tested on Kolena by building a workflow that meets its specific requirements.
If your use case meets any of the following conditions, we recommend building your own workflow:

- Your use case doesn't fit into one of the standard boxes defined as a built-in workflow (e.g. object detection)
- You use in-house or non-standard evaluation criteria
- You use exotic data types
- You want to plug in your own data visualizations
- You want full control over evaluation and metrics

When building a workflow you have control over every piece of the problem:

- [**Test Sample**](../quickstart/quickstart-building-a-workflow.md#test-sample-type): the input to your model.
  If your model is a standard single-image computer vision model, this might be locator of an image in an S3 bucket,
  e.g. `s3://my-bucket/image.png`.

  Any additional inputs required by your model, such as annotations produced by upstream models in your pipeline, are
  also included as a part of the test sample.

- [**Ground Truth**](../quickstart/quickstart-building-a-workflow.md#ground-truth-and-inference-types): the
  target against which your model is evaluated.

- [**Inference**](../quickstart/quickstart-building-a-workflow.md#ground-truth-and-inference-types): the
  prediction produced by your model. A model is a deterministic transformation from your test sample type into your
  inference type. Inferences are typically compared against ground truths to produce metrics.

- [**Metrics**](../quickstart/quickstart-building-a-workflow.md#step-2-defining-metrics): the metrics that
  describe your model's performance:

    - [Single-test-sample metrics](../quickstart/quickstart-building-a-workflow.md#test-sample-metrics), such as the loss
      calculated between a ground truth and inference object, the number of false positive predictions, etc.
    - [Test-case-level aggregate metrics](../quickstart/quickstart-building-a-workflow.md#test-case-metrics),
      such as precision, recall, accuracy, or any other aggregate metric computed across a population.
    - [Test-suite-level metrics](../quickstart/quickstart-building-a-workflow.md#test-suite-metrics) measuring
      performance across the different test cases in the test suite, e.g. penalizing performance variance across different
      subsets of the population.

- **Plots**: any plots visualizing your model's performance across a test case. Plots are entirely optional, and you can
  define as many as you see fit for your use case.

Building a workflow on Kolena is a straightforward process that involves defining data types and plugging in your
existing metrics computation functions.

See the [tutorial on building a workflow](../quickstart/quickstart-building-a-workflow.md) for a guided walk
through the process of testing your arbitrary ML problem on Kolena.


## Built-in Workflow: Face Recognition (1:1)

![Example Face Recognition (1:1) image pair from the Labeled Faces in the Wild dataset.](../.gitbook/assets/face-recognition-person.png)

Face Recognition (1:1) workflow is built to test models answering the question: _do these two images depict the same
person_? The terminology and methodology are adapted from the [NIST FRVT 1:1](https://pages.nist.gov/frvt/html/frvt11.html)
challenge.

The Face Recognition (1:1) workflow is also referred to as **Face Verification**.

#### Terminology

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
| **Baseline**                    | The test case(s) against which similarity score thresholds are computed ([see below](workflows.md#test-suite-baseline))      |

#### Test Suite Baseline

In the Face Recognition (1:1) workflow, similarity score thresholds are computed based on target false match rates over
a special section of your test suite known as the **baseline**.

In the simple standard case, you typically want the baseline to be the entire test suite. However, having control over
the baseline allows you to define test suites that answer questions like:

- How does my model perform on a certain population when the thresholds are computed using a different population (i.e.
  FMR/FNMR shift)?
- How does my model perform in different deployment conditions (e.g. low lighting, infrared) when using thresholds
  computed from standard conditions? (i.e. can my model generalize to unseen scenarios?)

#### Model Pipeline

The Face Recognition (1:1) workflow is built to accommodate standard face recognition model pipelines with the following
steps:

1. Face detection (using object detection model)
2. Face alignment (using landmark detection model)
3. Feature extraction (using embedding extraction model)
4. Feature comparison (using embeddings comparison algorithm)

#### Multiple Faces in a Single Image

The Face Recognition (1:1) workflow supports extraction of any number of faces from a given image during testing,
following the methodology of the [NIST FRVT 1:1 Verification](https://pages.nist.gov/frvt/html/frvt11.html)
specification section [Accuracy Consequences of Recognizing Multiple Faces in a Single Image](https://pages.nist.gov/frvt/html/slides/multiple_faces_single_image.pdf).

During testing, any number between zero (i.e. failure to enroll) and N embeddings may be uploaded for a given image.
When computing similarity scores between embeddings from two images in an image pair, all combinations of embeddings
extracted from the left and the right image in the pair are computed. When computing metrics, only the highest
similarity score between different embeddings in the image pair is considered.
