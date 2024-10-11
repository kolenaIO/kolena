# Example Integration: Instance Segmentation

This example integration uses the [COCO](https://cocodataset.org/#overview) dataset to demonstrate how to test
instance segmentation problems on Kolena. Only images with the
[Attribution 2.0](https://creativecommons.org/licenses/by/2.0/) license are included.

The Instance Segmentation example largely mirrors the [2D Object Detection example](../object_detection_2d/README.md),
using the same dataset, ground truths, and inferences, but leverages the `kolena.annotation.Polygon` type annotation
over `kolena.annotation.BoundingBox`, while computing similar metrics using IoU computations against these polygons.
For an approach that leverages masks for annotation, please see the [Semantic Segmentation example](../semantic_segmentation/README.md).

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for packaging and Python dependency management. To get started,
install project dependencies from [`pyproject.toml`](./pyproject.toml) by running:

```shell
uv sync
```

## Usage

All data for this example integration lives in the publicly accessible S3 bucket `s3://kolena-public-examples`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.com/installing-kolena/#initialization) for details.

This project defines two scripts that perform the following operations:

1. [`upload_dataset.py`](instance_segmentation/upload_dataset.py) uploads the [COCO](https://cocodataset.org/#overview)
dataset - only transportation relevant object annotations are used in this example.

```shell
$ uv run instance_segmentation/upload_dataset.py --help
usage: upload_dataset.py [-h] [--dataset DATASET]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  Optionally specify a custom dataset name to upload.
```

2. [`upload_results.py`](instance_segmentation/upload_results.py) uploads results for one of the following
models: `yolo_r`, `yolo_x`, `mask_rcnn`, `faster_rcnn`, `yolo_v4s`, and `yolo_v3`.

The `upload_results.py` script defines command line arguments to select which model to evaluate — run using the
`--help` flag for more information:

```shell
$ uv run instance_segmentation/upload_results.py --help
usage: upload_results.py [-h] [--dataset DATASET] {yolo_r,yolo_x,mask_rcnn,faster_rcnn,yolo_v4s,yolo_v3}

positional arguments:
  {yolo_r,yolo_x,mask_rcnn,faster_rcnn,yolo_v4s,yolo_v3}
                        Name of the model to test.

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Optionally specify a custom dataset name to test.
```

## Quality Standards Guide

Once the dataset and results have been uploaded to Kolena, visit [Kolena](https://app.kolena.com/redirect/) to
[explore the data and results](https://docs.kolena.com/dataset/quickstart/#step-3-explore-data-and-results).

Here are our [Quality Standards](https://docs.kolena.com/dataset/core-concepts/quality-standard/) recommendations
for Instance Segmentation:

### Metrics

1. [Precision](https://docs.kolena.com/metrics/precision)
2. [Recall](https://docs.kolena.com/metrics/recall)
3. [F1-score](https://docs.kolena.com/metrics/f1-score)

### Plots

1. Distribution: `datapoint.bounding_boxes[].label`
2. Distribution: `result.TP[].label`
3. Distribution: `result.Confused[].label`
4. Distribution: `result.FP[].label`
5. Distribution: `result.FN[].label`
6. `datapoint.brightness` vs. `mean(count_FN)`
7. `datapoint.brightness` vs. `mean(count_FP)`
8. `datapoint.brightness` vs. `mean(count_TP)`

### Test Cases

1. `datapoint.brightness`
