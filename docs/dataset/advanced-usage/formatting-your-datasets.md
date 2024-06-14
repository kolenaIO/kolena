---
icon: kolena/studio-16
---

# :kolena-studio-20: Data Formatting

Kolena has a set of powerful data visualization funcitonalities that enable teams to identify patterns and
generate actionable insights fast.

<div class="grid cards" markdown>
- [:kolena-properties-16: Fomatting data for Computer Vision](./dataset-formatting/computer-vision.md)

    ---

    Setup Datasets and model results for best experience for 2D, 3D object setection, Semantic segmentation
    and Video based ML problems.

- [:kolena-metrics-glossary-16: Fomratting data for Audio](./dataset-formatting/audio.md)

    ---
    Instruction on how to setup datasets and model results for audio based data.

- [:kolena-metrics-glossary-16: Fomratting data for NLP and LLM](./dataset-formatting/natural-language.md)

    ---
    Instruction on how to setup datasets and model results for NLP and LLM problems.

</div>

## remove the following
Kolena will look for the following fields when displaying datapoints:

| Field Name | Description                                                                                                                              |
|------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `locator`  | Url path to a file to be displayed, either a [cloud storage](../../connecting-cloud-storage/index.md) url or a http url that serves a file. |
| `text`     | Raw text input for text based models.                                                                                                    |

A locator needs to have correct extensions for the corresponding file type. For example an image should be in a format
such as `.jpg` or `.png`, whereas locators for audio data should be in forms like `.mp3` or `.wav`.

### datapoint visualizaiton types

Kolena allows you to visualize your datasets by use of the Studio. The studio experience depends on the type of data
relevant to your problem.

The first experience is the Gallery view which allows you to view your data in a grid. This is useful as you can see
chunks of your data (images, video, audio, text) and view results without having to view each datapoint individually.

The second experience is the Tabular view, used when your data is a set of columns and values.
An example of this is the [:kolena-widget-16: Rain Forcast ↗](https://github.com/kolenaIO/kolena/tree/trunk/examples/dataset/rain_forecast)
dataset.

### Kolena Assets

You can connect files to datapoints in Kolena by the use of [`asset`](../../reference/asset.md), which can be visualized
in the Studio when exploring datasets and results. Multiple assets can be attached to a single datapoint allowing you to
represent complex scenarios on Kolena. Assets are files stored in a cloud bucket or served at a URL.

The `s3://kolena-public-examples/LibriSpeech/raw/LibriSpeech.csv` csv contains data of following format:

| id                | audio                                                                                      | transcript             | word_count |
|-------------------|--------------------------------------------------------------------------------------------|------------------------|------------|
| 1272-128104-0014  | `s3://kolena-public-examples/LibriSpeech/data/dev-clean/1272/128104/1272-128104-0014.flac` | `by harry quilter m a` | 5          |

Here the audio column contains a locator but if uploaded as is, it would just be rendered as a text metadata field.
We need to use the `AudioAsset` annotation when uploading in order for the Audio file to be rendered as an asset.

```python
from kolena.asset import AudioAsset
from kolena.io import dataframe_to_csv
import pandas as pd


df = pd.read_csv("s3://kolena-public-examples/LibriSpeech/raw/LibriSpeech.csv", storage_options={"anon": True})
df["audio"] = df["audio"].apply(AudioAsset)
dataframe_to_csv(df, "audio-asset.csv")
```
Now the data in `audio-asset.csv` can be uploaded as a tabular dataset with audio assets attached to each row.
Any name can be used for the `audio` column in this example.

!!! example

    **2D Object Detection**


The following snippet shows how to format COCO data as a dataset within Kolena. As the input `.csv` file contains rows
for each bounding box within an image, we need to apply some transformations to the raw data.
This is done by creating a list of all bounding boxes for an image and then merging it with the metadata.
The produced `.csv` contains a column called ground_truths where the data is the same format as the above bounding boxes.

```python
from kolena.annotation import LabeledBoundingBox
from kolena.io import dataframe_to_csv
from collections import defaultdict
import pandas as pd

df = pd.read_csv(f"s3://kolena-public-examples/coco-2014-val/transportation/raw/coco-2014-val.csv",
                 storage_options={"anon": True})
image_to_boxes = defaultdict(list)
image_to_metadata = defaultdict(dict)

for record in df.itertuples():
    coords = (float(record.min_x), float(record.min_y)), (float(record.max_x), float(record.max_y))
    bounding_box = LabeledBoundingBox(*coords, record.label)
    image_to_boxes[record.locator].append(bounding_box)
    metadata = {
        "locator": record.locator,
        "height": record.height,
        "width": record.width,
        "date_captured": record.date_captured,
        "brightness": record.brightness,
    }
    image_to_metadata[record.locator] = metadata

df_boxes = pd.DataFrame(list(image_to_boxes.items()), columns=["locator", "ground_truths"])
df_metadata = pd.DataFrame.from_dict(image_to_metadata, orient="index").reset_index(drop=True)
df_merged = df_metadata.merge(df_boxes, on="locator")

dataframe_to_csv(df_merged, "processed.csv")
```

The file `processed.csv` can be uploaded through the [:kolena-dataset-16: Datasets](https://app.kolena.com/redirect/datasets)
page.

## Formatting Results

### Formatting results for Object Detection

For Object Detection problems, model results need to have the following columns
for the best experience. The values for each of the columns is a [`List[ScoredLabeledBoundingBox]`](../../reference/annotation.md#kolena.annotation.ScoredLabeledBoundingBox)

| Column Name              | Description                                         |
|--------------------------|-----------------------------------------------------|
| `matched_inference`      | Inferences that were matched to a ground truth.     |
| `unmatched_inference`    | Inferences that were not matched to a ground truth. |
| `unmatched_ground_truth` | Ground truths with no matching inference.           |

These columns are used to determine `True Postitives`, `False Positives`, and `False Negatives`.
These results can be formatted for upload with a similar process as above. This is done by adding the relevant list of
bounding boxes to the `matched_inference`, `unmatched_inference`, and `unmatched_ground_truth` columns for each image.
The `results.csv` created can be uploaded by opening the corresponding dataset from the
[:kolena-dataset-16: Datasets](https://app.kolena.com/redirect/datasets) page and navigating to the Studio section.

We have provided an [:kolena-widget-16: Object Detection (2D) ↗](https://github.com/kolenaIO/kolena/tree/trunk/examples/dataset/object_detection_2d)
example that shows how to take raw results and perform bounding box matching to produce the values mentioned above.

### To use compound metrics on the fly

The Kolena web application currently supports [`precision`](../../metrics/precision.md),
[`recall`](../../metrics/recall.md), [`f1_score`](../../metrics/f1-score.md),
[`accuracy`](../../metrics/accuracy.md), [`false_positive_rate`](../../metrics/fpr.md),
and [`true_negative_rate`](../../metrics/recall.md).

To leverage these, add the following columns to your CSV: `count_TP`, `count_FP`, `count_FN`, `count_TN`.

## Supported File Data Formats

The Kolena web application currently supports various file formats for both dataset uploads and model
results processing. The following table lists the supported file formats:

| Format    | Description                                              |
|-----------|----------------------------------------------------------|
| `.csv`     | Comma-separated values file, ideal for tabular data.     |
| `.parquet` | Apache Parquet format, efficient for columnar storage.   |
| `.jsonl`   | JSON Lines format, suitable for handling nested data.    |

**CSV Files**: Widely used for simple tabular datasets, CSV files are easy to generate and manipulate,
making them a popular choice for data scientists and developers.

**Parquet Files**: Offering efficient storage and fast retrieval, Parquet files are optimal for
handling large datasets with a significant number of columns.

**JSON Lines (JSONL) Files**: Each line in a JSONL file is a complete JSON object, making this format
ideal for datasets with complex or nested data structures.

When preparing your dataset or model results files for upload, ensure that they conform to one of these
supported file formats to guarantee compatibility with Kolena's data processing capabilities.
