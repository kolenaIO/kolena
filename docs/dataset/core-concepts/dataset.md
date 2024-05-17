---
icon: kolena/dataset-16
search:
  boost: 2
---

# :kolena-dataset-20: Dataset

A **dataset** is a version-controlled collection of datapoints.
Datapoints within a given version are immutable.
This immutability ensures the integrity and traceability of the data used in model evaluation.

## Datapoints

Datapoints can be thought of as the "units" of model evaluation.
Datapoints may represent a variety of media including images, video, documents, text, or even 3D point clouds.
Datapoints can have properties, known as "fields" which may represent primitive values like strings or numbers,
nested objects, or media assets and annotations like [`bounding boxes`][kolena.annotation.BoundingBox].

Key characteristics of datapoints include:

- **Flexible Data Structure**:
  Datapoints are generic data containers, allowing for customization.
  Datapoints can represent whatever "unit" of testing is relevant for your problem.
  In computer vision, a datapoint may represent an image with associated bounding box ground truths.
  For language models, a datapoint may represent text prompts.

- **Immutability**: Datapoints within a given dataset version cannot be altered.
  Any update to a datapoint results in the creation of a new dataset version.

- **Traceability**: Updates to datapoints between versions are recorded for auditing purposes.

- **Isolation Between Datasets**:
  Datapoints are exclusive to the dataset they belong to and are not shared across different datasets.
  This exclusivity ensures clear demarcation and management of data.

### How Are Datasets Represented?

Datasets are represented using tables, with each row representing a distinct datapoint.
One column, or a combination of columns, serve as the primary key to this table and uniquely
identify each datapoint. These are known as the **ID Field(s)**.

Likewise, Model Results for a dataset are also represented using tables.
Model Results must contain the same ID Fields as your dataset so that
Kolena can associate model results with the appropriate datapoints.

??? note "Selecting ID Fields"
    We recommend
    that your ID fields be convenient to generate and pass around with your model results. Typically, this means selecting
    a single short string or integer field as the ID.

    For more information on how to specify your ID fields, see the relevant documentation on
    [formatting your datasets](../formatting-your-datasets.md#what-defines-a-datapoint).

Datasets can be created using CSV or Parquet files, or more generally from Pandas DataFrames.
See our detailed documentation to learn more about how to [format your datasets](../formatting-your-datasets.md).
