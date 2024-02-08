
# Formatting your Datasets

## What is a Dataset

A **dataset** is a structured assembly of datapoints, designed for model evaluation.
Each datapoint in a dataset is a comprehensive unit that combines data
traditionally segmented into test samples, ground truth, and metadata.

### What defines a Datapoint

Conceptually a datapoint is a set of inputs that you would want to test on your models.
Consider a single row of data for a classification problem with the following columns <br>

| locator                    | ground_truth |
|---------------------------------------------------------------|----------------------------------------|
| s3://kolena-public-examples/cifar10/data/horse0000.png        | horse                                  |

An `id_field` is required in order to differentiate between datapoints.
By default the `locator` or  `text` fields are used if present in your `dataframe` or `.csv`.
You can specify other fields as the id_field both when importing via the Web App or the SDK.

A `locator` is a url path to a file that will be displayed in the platform. Locators can either be urls for common
cloud storage providers, a local file path or a http url that serves a file. A locator needs to have correct extensions
for the corresponding type of file. For example an image should be in a format such as `.jpg` or `.png`,
whereas locators for audio data should be in forms like `.mp3` or `.wav`.

For text-based models the `text` field contains the raw text input for the models.

Metadata and other additional fields can be added to datasets by adding a column to the `.csv` and providing values for
datapoints where applicable. For example `image_height` and `image_width` are useful metadata for image datasets and
fields like `word_count` are useful for text datasets.

## How are Datasets viewed

Kolena allows you to visualize your datasets by use of the Studio. The studio experience depends on the type of data
relevant to your problem.

The first experience is the Gallery view which allows you to view your data in a grid. This is useful as you can see
chunks of your data (images, video, audio, text) and view results without having to view each datapoint individually.

The second experience is the Tabular view, used when your data is a set of columns and values. For example rain data
for a set of locations over a period of time).

In order to use the Gallery view you just need to have the `locator` or `text` fields specfied in the dataset.

## Enriching you Dataset expirience

### Kolena Assets

You can connect files to datapoints in Kolena by the use of Assets, which can be visualized when exploring datasets,
test cases or results. Multiple assets can be attached to a single datapoint allowing you to represent complex scenarios
on Kolena.  Assets are files stored in a cloud bucket or served at a URL.

| Asset Type  | Description |
|-------------|----------------------------|
| ImageAsset  | Useful if your data is modeled as multiple related images |
| BinaryAsset | Useful if you want to attach any segmentation or bitmap masks |
| AudioAsset  | Useful if you want to attach an audio file |
| VideoAsset  | Useful if you want to attach a video file |
| PointCloudAsset | Useful for attaching 3D point cloud data |

### Kolena Annotations

Kolena allows you to visualize overlays on top of datapoints through the use of `Annotations`.
These annotations are visible on both the Gallery view for groups of datapoints and for individual datapoints.

| Asset Type  | Description |
|-------------|----------------------------|
| BoundingBox  | Used to overlay bounding boxes (including confidence scores and labels) on top of images. |
| SegmentationMask | Used to overaly raster segmentation maps on top of images. |

### Structured Data

When uploading `.csv` files for datasets that contain annotations, assets or nested values in a column use the
`dataframe_to_csv()` function provided by Kolena to save a `.csv` instead of `pandas.to_csv()`.

In order to add structured data like a list of `BoundingBoxes` to your dataset via the sdk all you need to do is have
field with a list of objects (Works for Kolena annotations) in your dataframe.

A snippet like the following:

```python
from kolena.annotation import BoundingBox
from kolena.io import dataframe_to_csv

df = pd.read_csv(f"s3://kolena-public-examples/300-W/coco-2014-val/coco-2014-val.csv", storage_options={"anon": True})
image_to_boxes: Dict[str, List[BoundingBox]] = defaultdict(list)

for record in df_metadata_csv.itertuples():
    coords = (float(record.min_x), float(record.min_y)), (float(record.max_x), float(record.max_y))
    bounding_box = BoundingBox(*coords)
    image_to_boxes[record.locator].append(bounding_box)



df["bounding_boxes"] = [BoundingBox(top_left=(df["min_x"], df["min_y"]),
                                    bottom_left=(df["max_x"], df["max_y"])) for points in df["points"]]
dataframe_to_csv(df, "processed.csv")
```

## Formatting results for Object Detection

For Object Detection problems, model results need to have the following columns
for the best experience.

If you already have this data and want to upload a .csv directly then you need to make sure the results
are configured correctly.

| Column name | Description                                           | Type |
|-------------|-------------------------------------------------------|------|
| matched_inference  | inferences that were matched to a ground_truth        |  List[ScoredLabeledBoundingBoxes]|
| unmatched_inference | inferences that were not matched to a ground_truth| List[ScoredLabeledBoundingBoxes]|
| unmatched_ground_truth | inferences that were not matched to a ground_truth| List[ScoredLabeledBoundingBoxes]|

These columns are used to determine `True Postitives`  `False Positives` and `False Negatives`.

We have provided an example that shows how to take raw results (`BoundingBoxes`, `confidence_scores` and `labels` and
perform bounding box matching to produce the values mentioned above.

## Configuring Thumbnails

Add a field called thumbnail_locator to the data, where the value points to a compressed version of the data.
This image should have the same dimensions as the original image to ensure that overlays are rendered correctly.
This will significantly improve the performance of Studio - images load much faster when filtering, sorting or using
the embedding sort.
