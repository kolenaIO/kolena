---
icon: kolena/studio-16
---

# :kolena-studio-20: Formatting your Datasets

## What is a Dataset

A [dataset](../dataset/core-concepts/dataset.md) is a structured assembly of datapoints, designed for model evaluation.
Each datapoint in a dataset is a comprehensive unit that combines data
traditionally segmented into test samples, ground truth, and metadata.

### What defines a Datapoint

Conceptually a datapoint is a set of inputs that you would want to test on your models.
Consider a single row of data for a classification problem with the following columns:

| locator                    | ground_truth |
|---------------------------------------------------------------|----------------------------------------|
| s3://kolena-public-examples/cifar10/data/horse0000.png        | horse                                  |

An `id_field` is required in order to differentiate between datapoints.
By default the `locator` or  `text` fields are used if present in your dataset, and other fields
can be specified when importing via the Web App or the SDK.

A `locator` is a url path to a file that will be displayed on the platform and can either be a
[cloud storage](../connecting-cloud-storage/index.md) url or a http url that serves a file.
A locator needs to have correct extensions for the corresponding file type. For example an image should be in a format
such as `.jpg` or `.png`, whereas locators for audio data should be in forms like `.mp3` or `.wav`.

| Data Type    | Supported file formats                                                                |
|--------------|---------------------------------------------------------------------------------------|
| Image        | `jpg`,`jpeg`, `png`, `gif`, `bmp` and other web browser supported image types.        |
| Audio        | `flac`, `mp3`, `wav`, `acc`, `ogg`, `ra` and other web browser supported audio types. |
| Video        | `mov`, `mp4`, `mpeg`, `avi` and other web browser supported video types.              |

For text-based models the `text` field contains the raw text input for the models.

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

In order to use the Gallery view you just need to have the `locator` or `text` fields specfied in the dataset.

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

### Kolena Annotations

Kolena allows you to visualize overlays on top of datapoints through the use of[`annotation`](../reference/annotation.md).
These annotations are visible on both the Gallery view for groups of datapoints and for individual datapoints.

| Annotation Type                                                                      | Description |
|--------------------------------------------------------------------------------------|----------------------------|
| [`BoundingBox`](../reference/annotation.md#kolena.annotation.BoundingBox)            | Used to overlay bounding boxes (including confidence scores and labels) on top of images. |
| [`SegmentationMask`](../reference/annotation.md#kolena.annotation.SegmentationMask)  | Used to overaly raster segmentation maps on top of images. |

### Structured Data

When uploading `.csv` files for datasets that contain annotations, assets or nested values in a column use the
[`dataframe_to_csv()`](../reference/io.md#kolena.io.dataframe_to_csv) function provided by Kolena to save a `.csv` file
instead of [`pandas.to_csv()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html).

In order to add data like a list of `BoundingBox` objects to your dataset via the sdk all you need to do is have
field with a list of objects in your dataframe.

A snippet like the following:

```python
from kolena.annotation import BoundingBox
from kolena.io import dataframe_to_csv
from collections import defaultdict
import pandas as pd

df = pd.read_csv(f"s3://kolena-public-examples/coco-2014-val/raw/coco-2014-val.csv", storage_options={"anon": True})
image_to_boxes = defaultdict(list)
image_to_metadata = defaultdict(dict)

for record in df.itertuples():
    coords = (float(record.min_x), float(record.min_y)), (float(record.max_x), float(record.max_y))
    bounding_box = BoundingBox(*coords)
    image_to_boxes[record.locator].append(bounding_box)
    metadata = {
            "locator": str(record.locator),
            "height": float(record.height),
            "width": float(record.width),
            "date_captured": str(record.date_captured),
            "brightness": float(record.brightness),
        }
    image_to_metadata[record.locator] = metadata

df_boxes = pd.DataFrame(list(image_to_boxes.items()), columns=["locator", "ground_truths"])
df_metadata = pd.DataFrame.from_dict(image_to_metadata, orient="index").reset_index(drop=True)
df_boxes.merge(df_metadata, on="locator")
dataframe_to_csv(df_boxes, "processed.csv")
```

### Formatting results for Object Detection

For Object Detection problems, model results need to have the following columns
for the best experience. The values for each of the columns is a [`List[ScoredLabeledBoundingBox]`](../reference/annotation.md#kolena.annotation.ScoredLabeledBoundingBox)

| Column Name            | Description                                         |
|------------------------|-----------------------------------------------------|
| matched_inference      | Inferences that were matched to a ground_truth.     |
| unmatched_inference    | Inferences that were not matched to a ground_truth. |
| unmatched_ground_truth | Inferences that were not matched to a ground_truth. |

These columns are used to determine `True Postitives`  `False Positives` and `False Negatives`.

We have provided an [:kolena-widget-16: Object Detection (2D) ↗](https://github.com/kolenaIO/kolena/tree/trunk/examples/dataset/object_detection_2d)
that shows how to take raw results and perform bounding box matching to produce the values mentioned above.

### To use compound metrics on the fly

The Kolena web application currently supports [`precision`](../metrics/precision.md),
[`recall`](../metrics/recall.md), [`f1_score`](../metrics/f1-score.md),
[`accuracy`](../metrics/accuracy.md), [`false_positive_rate`](../metrics/fpr.md),
and [`true_negative_rate`](../metrics/recall.md).

To leverage these, add the following columns to your CSV: `count_TP`, `count_FP`, `count_FN`, `count_TN`.

### Configuring Thumbnails

Add a field called thumbnail_locator to the data, where the value points to a compressed version of the data.
This image should have the same dimensions as the original image to ensure that overlays are rendered correctly.
This will significantly improve the performance of Studio - images load much faster when filtering, sorting or using
the [embedding](../dataset/advanced-usage/set-up-natural-language-search.md) sort.