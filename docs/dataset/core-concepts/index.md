---
icon: kolena/area-of-interest-16
---

# :kolena-area-of-interest-20: Core Concepts

In this section, we explore Kolena's core concepts, focusing on the key features that facilitate model evaluation and
testing. For a quick overview, refer to the [Quickstart Guide](../quickstart.md).

## :kolena-dataset-20: Dataset

A **dataset** is a structured assembly of datapoints, designed for model evaluation.
This structure is immutable, meaning once a datapoint is added,
it cannot be altered without creating a new version of the dataset.
This immutability ensures the integrity and traceability of the data used in testing models.

### Datapoints

Datapoints are versatile and immutable objects with the following key characteristics:

- **Unified Object Structure**:
  Datapoints are singular, grab-bag objects that can embody various types of data,
  including images, as indicated by the presence of a data_type field.

- **Immunity to Change**: Once a datapoint is added to a dataset, it cannot be altered.
  Any update to a datapoint results in the creation of a new datapoint, and this action consequently versions the dataset.

- **Exclusive Association with Datasets**:
  Datapoints are exclusive to the dataset they belong to and are not shared across different datasets.
  This exclusivity ensures clear demarcation and management of data within specific datasets.

- **Role in Data Ingestion**: Datapoints play a central role in the data ingestion process.
  They are represented in a DataFrame structure with special treatment for certain columns like `locator` and `text`.

- **Extension of Data Classes**: Datapoints extend data classes, allowing for flexibility and customization.
  For instance, they can include annotation objects like [`BoundingBox`][kolena.annotation.BoundingBox],
  and these objects can be further extended as needed.

??? "How to generate datapoints"
    You can structure your dataset as a CSV file. Each row in the file should represent a distinct datapoint.
    For complete information on creating datasets, visit [formatting your datasets](../advanced-usage/formatting-your-datasets.md).

## :kolena-quality-standard-20: Quality Standard

A **Quality Standard** tracks a standardized process for how a team evaluates a model's performance on a dataset.
Users may define and manage quality standards for a dataset in the Kolena web application using the
`Quality Standards` tab.

A Quality Standard is composed of [Test Cases](#test-cases) and [Metrics](#metrics).

### Test Cases

Test cases allow users to evaluate their datasets at various levels of division, providing visibility into how models
perform at differing subsets of the full dataset, and mitigating failures caused by
[hidden stratifications](https://www.kolena.com/blog/best-practices-for-ml-model-testing).

Kolena supports easy test case creation through dividing a dataset along categorical or numeric datapoint properties.
For example, if you have a dataset with images of faces of individuals, you may wish to create a set of test cases that
divides your dataset by `datapoint.race` (categorical) or `datapoint.age` (numeric).

The quickstart guide provides a more hands-on example of
[defining test cases](../quickstart.md/#define-test-cases).

### Metrics

Metrics describe the criteria used to evaluate the performance of a model and compare it with other models over a given
dataset and its test cases.

Kolena supports defining metrics by applying standard aggregations over datapoint level results or by leveraging
common machine learning aggregations, such as [Precision](../../metrics/precision.md) or
[F1 Score](../../metrics/f1-score.md). Once defined, users may also specify highlighting for metrics, indicating if
`Higher is better`, or if `Lower is better`.

The datasets quickstart provides a more hands-on example of
[defining metrics](../quickstart.md/#define-metrics). For more details on out-of-the-box and custom metrics visit [Task Metrics](../advanced-usage/task-metrics.md)

### Model Comparison

Once you've defined your test cases and metrics, you can view and compare model results in the `Quality Standards` tab,
which provides a quick and standardized high level overview of which models perform best over your different test cases.

For step-by-step instructions, take a look at the quickstart for
[model comparison](../quickstart.md/#step-5-compare-models).

### Debugging

The `Debugger` tab of a dataset allows users to experiment with test cases and metrics without saving them off to the
team level quality standards. This allows users to search for meaningful test cases and experiment with different
metrics with the confidence that they can safely save these updated values to their quality standards when comfortable,
without the risk of accidentally replacing what the team has previously defined. This also provides a view for
visualizing results and relations in plots.

For step-by-step instructions, take a look at the quickstart for
[results exploration](../quickstart.md/#step-3-explore-data-and-results).
