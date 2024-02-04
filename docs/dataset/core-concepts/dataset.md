---
icon: kolena/dataset-16
search:
  boost: 2
---

# :kolena-dataset-20: Dataset

A **dataset** is a structured assembly of datapoints, designed for model evaluation.
Each datapoint in a dataset is a comprehensive unit that combines data
traditionally segmented into test samples, ground truth, and metadata.
This structure is immutable, meaning once a datapoint is added,
it cannot be altered without creating a new version of the dataset.
This immutability ensures the integrity and traceability of the data used in testing models.

## Datapoints

Datapoints are integral components within the dataset structure used for evaluating models.
They are versatile and immutable objects that encompass the role traditionally played by
test samples, ground truth, and metadata. Key characteristics of datapoints include:

- **Unified Object Structure**:
  Datapoints replace the need for separate entities like test samples and ground truth.
  They are singular, grab-bag objects that can embody various types of data,
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

In essence, datapoints in this context are versatile,
immutable data units that are exclusively associated with a specific dataset,
playing a crucial role in model evaluation by
encapsulating various types of data and annotations within a unified object structure.

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

### How to shape a CSV file for ingestion

#### For multimedia datapoints

Add a `locator` column in the CSV to view multimedia datapoints in the web application.

#### For text-focused datapoints

Add a `text` column in the CSV to view text datapoints in the web application.

#### To incorporate [`annotation`](../../reference/annotation.md) and [`asset`](../../reference/asset.md) into CSV

Consider the following Python snippet:

```python
from kolena.annotation import Keypoints
from kolena.io import dataframe_to_csv

df = pd.read_csv(f"s3://kolena-public-examples/300-W/raw/300-W.csv", storage_options={"anon": True})
df["face"] = [Keypoints(points=json.loads(points)) for points in df["points"]]
dataframe_to_csv(df, "processed.csv")
```

#### To use compound metrics on the fly

The Kolena web application currently supports [`precision`](../../metrics/precision.md),
[`recall`](../../metrics/recall.md), [`f1_score`](../../metrics/f1-score.md),
[`accuracy`](../../metrics/accuracy.md), [`false_positive_rate`](../../metrics/fpr.md),
and [`true_negative_rate`](../../metrics/recall.md).

To leverage these, add the following columns to your CSV: `count_TP`, `count_FP`, `count_FN`, `count_TN`.
