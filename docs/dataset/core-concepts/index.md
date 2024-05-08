---
icon: kolena/area-of-interest-16
hide:
  - toc
---

# :kolena-area-of-interest-20: Core Concepts

In this section, we explore Kolena's core concepts, focusing on the key features that facilitate model evaluation and
testing. For a quick overview, refer to the [Quickstart Guide](../quickstart.md).

# :kolena-dataset-20: Dataset

A **dataset** is a structured assembly of datapoints, designed for model evaluation.
This structure is immutable, meaning once a datapoint is added,
it cannot be altered without creating a new version of the dataset.
This immutability ensures the integrity and traceability of the data used in testing models.

## Datapoints

Datapoints are integral components within the dataset structure used for evaluating models.
They are versatile and immutable objects. Key characteristics of datapoints include:

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

### How to generate datapoints?

Structure your dataset as a *CSV file*. Each row in the file should represent a distinct datapoint.

- **Reserved Columns**: for models that process images, audio, or video,
  include a `locator` column with valid URL paths to the respective files.
  For text-based models, include a `text` column that contains the input text data directly in the CSV.

- **Additional Fields**: Include relevant metadata depending on the data type.
  For instance, image datasets might have metadata like `image_width`, `image_height`, etc.
  Similarly, other data types can have their respective metadata fields that are useful for model processing.

- **Data Consistency and Format**: It's crucial to maintain data consistency.
  URLs should be correctly encoded, text should be properly formatted,
  and numerical values should adhere to their respective formats.

- **Data Accessibility**: Ensure the data, especially if linked through URLs, is accessible for processing.
  In the case of cloud storage, appropriate permissions should be in place to allow access.

### ID Fields

When you upload your dataset, you will need to specify one or more ID fields. These fields should form a primary
key for the dataset. In other words, each datapoint should have a distinct combination of values in the specified ID
field(s).

Kolena uses the ID field(s) in your dataset to associate your model results with the appropriate datapoints, and
you will need to include the ID field(s) as fields in any model results you upload. For this reason, we recommend
that your ID fields be convenient to generate and pass around with your model results. Typically, this means selecting
a single short string or integer field as the ID.

For more information on how to specify your ID fields, see the relevant documentation on
[formatting your datasets](../formatting-your-datasets.md#what-defines-a-datapoint).

# :kolena-quality-standard-20: Quality Standard

A **Quality Standard** tracks a standardized process for how a team evaluates a model's performance on a dataset.
Users may define and manage quality standards for a dataset in the Kolena web application from a that dataset's
`Quality Standards` tab. 

A Quality Standard is composed of [Test Cases](#test-cases) and [Metrics](#metrics).

## Test Cases

Test cases allow users to evaluate their datasets at various levels of division, providing visibility into how models
perform at differing subsets of the full dataset, and mitigating failures caused by
[hidden stratifications](https://www.kolena.com/blog/best-practices-for-ml-model-testing).

Kolena supports easy test case creation through dividing a dataset along categorical or numeric datapoint properties.
For example, if you have a dataset with images of faces of individuals, you may wish to create a set of test cases that
divides your dataset by `datapoint.race` (categorical) or `datapoint.age` (numeric).

The quickstart guide provides a more hands-on example of
[defining test cases](../quickstart.md/#define-test-cases).

## Metrics

Metrics describe the criteria used to evaluate the performance of a model and compare it with other models over a given
dataset and its test cases.

Kolena supports defining metrics by applying standard aggregations over datapoint level results or by leveraging
common machine learning aggregations, such as [Precision](../../metrics/precision.md) or
[F1 Score](../../metrics/f1-score.md). Once defined, users may also specify highlighting for metrics, indicating if
`Higher is better`, or if `Lower is better`.

The datasets quickstart provides a more hands-on example of
[defining metrics](../quickstart.md/#define-metrics).

## Model Comparison

Once you've defined your test cases and metrics, you can view and compare model results in the `Quality Standards` tab,
which provides a quick and standardized high level overview of which models perform best over your different test cases.

For step-by-step instructions, take a look at the quickstart for
[model comparison](../quickstart.md/#step-5-compare-models).

## Debugging

The `Debugger` tab of a dataset allows users to experiment with test cases and metrics without saving them off to the
team level quality standards. This allows users to search for meaningful test cases and experiment with different
metrics with the confidence that they can safely save these updated values to their quality standards when comfortable,
without the risk of accidentally replacing what the team has previously defined. This also provides a view for
visualizing results and relations in plots.

For step-by-step instructions, take a look at the quickstart for
[results exploration](../quickstart.md/#step-3-explore-data-and-results).