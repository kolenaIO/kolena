---
icon: kolena/studio-16
---

# :kolena-studio-20: Formatting your Datasets

## What is a Dataset

A [dataset](../dataset/core-concepts/dataset.md) is a structured assembly of datapoints, designed for model evaluation.
Each datapoint in a dataset is a comprehensive unit that combines data traditionally segmented into test samples,
ground truth, and metadata.

### What defines a Datapoint

Conceptually, a datapoint is a set of inputs that you would want to test on your models.
Consider a single row within the [:kolena-widget-16: Classification (CIFAR-10) ↗](https://github.com/kolenaIO/kolena/tree/trunk/examples/dataset/classification)
dataset with the following columns:

| locator                    | ground_truth | image_brightness |   image_contrast |
|---------------------------------------------------------------|--------------|----------|-----|
| `s3://kolena-public-examples/cifar10/data/horse0000.png`        | horse        |     153.994     |    84.126  |

From this you can see that image `horse0000.png` has the ground_truth classification of `horse`,
and has brightness and contrast data.

When uploading a dataset to Kolena, it is important to be able to differentiate between each datapoint. This is
accomplished by configuring an `id_field` - a unique identifier for a datapoint. You can select any field that is
unique across your data, or generate one if no unique identifiers exist for your dataset. Below are some common patterns
for generating/selecting a unique identifier if your data does not have a natural ID field:

- If your datapoints contain a `locator` field pointing to the external files representing your model inputs, the `locator`
  field is usually used as the ID field.
- For datapoints with a `text` field for text-based models, we recommend either generating and saving a UUID for each
  datapoint or generating a hash of the `text` field to use as the ID field. You can also use the `text` field itself
  as the ID field.
- For other kinds of datapoints, we recommend generating and saving a UUID for each datapoint to use as the ID field.

Kolena will attempt to infer common `id_field`s (eg. `locator`, `text`) based on what is present in the dataset during import.
This can be overridden by explicitly declaring id fields when importing via the Web App from the [:kolena-dataset-16: Datasets](https://app.kolena.com/redirect/datasets)
page, or the SDK by using the [`upload_dataset`](../reference/dataset/index.md#kolena.dataset.dataset.upload_dataset)
function.

Kolena will look for the following fields when displaying datapoints:

| Field Name | Description                                                                                                                              |
|------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `locator`  | Url path to a file to be displayed, either a [cloud storage](../connecting-cloud-storage/index.md) url or a http url that serves a file. |
| `text`     | Raw text input for text based models.                                                                                                    |

A locator needs to have correct extensions for the corresponding file type. For example an image should be in a format
such as `.jpg` or `.png`, whereas locators for audio data should be in forms like `.mp3` or `.wav`.

#### Locator Support Matrix

| Data Type      | Supported file formats                                                                |
|----------------|---------------------------------------------------------------------------------------|
| Image          | `jpg`, `jpeg`, `png`, `gif`, `bmp` and other web browser supported image types.       |
| Audio          | `flac`, `mp3`, `wav`, `acc`, `ogg`, `ra` and other web browser supported audio types. |
| Video          | `mov`, `mp4`, `mpeg`, `avi` and other web browser supported video types.              |
| Document       | `txt` and `pdf` files.                                                                |
| Point Cloud    | `pcd` files.                                                                          |

Metadata and other additional fields can be added to datasets by adding a column to the `.csv` and providing values for
datapoints where applicable. For example `image_height` and `image_width` may be useful metadata for image datasets and
fields like `word_count` may be useful for text datasets.

## How are Datasets viewed

Kolena allows you to visualize your datasets by use of the Studio. The studio experience depends on the type of data
relevant to your problem.

The first experience is the Gallery view which allows you to view your data in a grid. This is useful as you can see
chunks of your data (images, video, audio, text) and view results without having to view each datapoint individually.

The second experience is the Tabular view, used when your data is a set of columns and values.
An example of this is the [:kolena-widget-16: Rain Forcast ↗](https://github.com/kolenaIO/kolena/tree/trunk/examples/dataset/rain_forecast)
dataset.

In order to use the Gallery view you will need to have the `locator` or `text` fields specified in the dataset.

## Enriching your Dataset experience

### Kolena Assets

You can connect files to datapoints in Kolena by the use of [`asset`](../reference/asset.md), which can be visualized
in the Studio when exploring datasets and results. Multiple assets can be attached to a single datapoint allowing you to
represent complex scenarios on Kolena. Assets are files stored in a cloud bucket or served at a URL.

| Asset Type                                                              | Description                                                    |
|-------------------------------------------------------------------------|----------------------------------------------------------------|
| [`ImageAsset`](../reference/asset.md#kolena.asset.ImageAsset)           | Useful if your data is modeled as multiple related images.     |
| [`BinaryAsset`](../reference/asset.md#kolena.asset.BinaryAsset)         | Useful if you want to attach any segmentation or bitmap masks. |
| [`AudioAsset`](../reference/asset.md#kolena.asset.AudioAsset)           | Useful if you want to attach an audio file.                    |
| [`VideoAsset`](../reference/asset.md#kolena.asset.VideoAsset)           | Useful if you want to attach a video file.                     |
| [`PointCloudAsset`](../reference/asset.md#kolena.asset.PointCloudAsset) | Useful for attaching 3D point cloud data.                      |

The [:kolena-widget-16: Automatic Speech Recognition ↗](https://github.com/kolenaIO/kolena/tree/trunk/examples/dataset/automatic_speech_recognition)
example showcases how `AudioAsset`s can be attached to datapoints.

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

### Kolena Annotations

Kolena allows you to visualize overlays on top of datapoints through the use of[`annotation`](../reference/annotation.md).
These annotations are visible on both the Gallery view for groups of datapoints and for individual datapoints.

| Annotation Type                                                                      | Description                                                                               |
|--------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| [`BoundingBox`](../reference/annotation.md#kolena.annotation.BoundingBox)            | Used to overlay bounding boxes (including confidence scores and labels) on top of images. |
| [`SegmentationMask`](../reference/annotation.md#kolena.annotation.SegmentationMask)  | Used to overlay raster segmentation maps on top of images.                                |

### Structured Data

Consider a `.csv` file containing ground truth data in the from of bounding boxes for an Object Detection problem.

| locator                                                                       | label      | min_x     | max_x  | min_y | max_y   |
|-------------------------------------------------------------------------------|------------|-----------|--------|-------|---------|
| s3://kolena-public-examples/coco-2014-val/data/COCO_val2014_000000369763.jpg | motorcycle | 270.77    | 621.61 | 44.59 |  254.18  |
| s3://kolena-public-examples/coco-2014-val/data/COCO_val2014_000000369763.jpg | car        | 538.03    | 636.85 | 8.86  | 101.93  |
| s3://kolena-public-examples/coco-2014-val/data/COCO_val2014_000000369763.jpg | trunk      | 313.02    | 553.98 | 12.01 | 99.84   |

The first bounding box for the image is `(270.77, 44.59), (621.61,  254.18)`. To represent this within Kolena use the
[`LabeledBoundingBox`](../reference/annotation.md#kolena.annotation.LabeledBoundingBox) annotation. If you want to ignore
labels the base [`BoundingBox`](../reference/annotation.md#kolena.annotation.BoundingBox) can be used.

This looks like:
```python
from kolena.annotation import LabeledBoundingBox
bbox = LabeledBoundingBox(top_left=(270.77, 44.59), bottom_right=(621.61,  254.18), label="motorcycle")
```
When viewing a bounding box within python the format is:
```
LabeledBoundingBox(top_left=(270.77, 44.59), bottom_right=(621.61, 254.18), label="motorcycle", width=350.84,
 height=209.59, area=73532.5556, aspect_ratio=1.67)
```

A single bounding box would be serialized as the following JSON string within a `.csv` file:

```
{""top_left"": [270.77, 44.59], ""bottom_right"": [621.61, 254.18], ""width"": 350.84, ""height"": 209.59,
 ""area"": 73532.5556, ""aspect_ratio"": 1.67, ""label"": ""motorcycle"",  ""data_type"": ""ANNOTATION/BOUNDING_BOX""},
```

The above example has multiple objects within a single image, which is represented in Kolena as a list of bounding boxes.

For example:
```python
from kolena.annotation import LabeledBoundingBox
bboxes = [
    LabeledBoundingBox(top_left=(270.77, 44.59), bottom_right=(621.61, 254.18), label="motorcycle"),
    LabeledBoundingBox(top_left=(538.03, 8.86), bottom_right=(636.85, 101.93), label="car"),
    LabeledBoundingBox(top_left=(313.02, 12.01), bottom_right=(553.98, 99.84), label="trunk"),
]
```
This would be represented within a `.csv` file as shown below. Note this will be a single line,
but is shown here as multiple lines for formatting.
```
"[{""top_left"": [270.77, 44.59], ""bottom_right"": [621.61, 254.18], ""width"": 350.84, ""height"": 209.59,
 ""area"": 73532.5556, ""aspect_ratio"": 1.67, ""label"": ""motorcycle"", ""data_type"": ""ANNOTATION/BOUNDING_BOX""},
  {""top_left"": [538.03, 8.86], ""bottom_right"": [636.85, 101.93], ""width"": 98.82, ""height"": 93.07,
   ""area"": 9197.1774, ""aspect_ratio"": 1.062, ""label"": ""car"", ""data_type"": ""ANNOTATION/BOUNDING_BOX""},
  {""top_left"": [313.02, 12.01], ""bottom_right"": [553.98, 99.84], ""width"": 240.96,
   ""height"": 87.83, ""area"": 21163.5168, ""aspect_ratio"": 2.743, ""label"": ""trunk"", ""data_type"": ""ANNOTATION/BOUNDING_BOX""}]"
```

When uploading `.csv` files for datasets that contain annotations, assets or nested values in a column use the
[`dataframe_to_csv()`](../reference/io.md#kolena.io.dataframe_to_csv) function provided by Kolena to save a `.csv` file
instead of [`pandas.to_csv()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html).
`pandas.to_csv` does not serialize Kolena annotation objects in a way that is compatible with the platform.

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

### Configuring Thumbnails

In order to improve the loading performance of your image data, you can upload compressed versions of the image
with the same dimensions as thumbnails. This results in an improved Studio experience due to faster image loading
when filtering, sorting or using [embedding](../dataset/advanced-usage/set-up-natural-language-search.md) sort.

Thumbnails are configured by adding a field called `thumbnail_locator` to the data, where the value points
to a compressed version of the `locator` image.

If you wanted to add a thumbnail to the classification data shown above it would look like:

| locator                    | thumbnail_locator                                                  | ground_truth | image_brightness |   image_contrast |
|---------------------------------------------------------------|--------------------------------------------------------------------|--------------|----------|-----|
| `s3://kolena-public-examples/cifar10/data/horse0000.png`        | `s3://kolena-public-examples/cifar10/data/thumbnail/horse0000.png` | horse        |     153.994     |    84.126  |

## Formatting Results

### Formatting results for Object Detection

For Object Detection problems, model results need to have the following columns
for the best experience. The values for each of the columns is a [`List[ScoredLabeledBoundingBox]`](../reference/annotation.md#kolena.annotation.ScoredLabeledBoundingBox)

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

The Kolena web application currently supports [`precision`](../metrics/precision.md),
[`recall`](../metrics/recall.md), [`f1_score`](../metrics/f1-score.md),
[`accuracy`](../metrics/accuracy.md), [`false_positive_rate`](../metrics/fpr.md),
and [`true_negative_rate`](../metrics/recall.md).

To leverage these, add the following columns to your CSV: `count_TP`, `count_FP`, `count_FN`, `count_TN`.
