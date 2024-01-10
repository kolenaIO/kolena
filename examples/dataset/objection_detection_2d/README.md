# Example Integration: Object Detection (2D)

This example integration uses the [COCO](https://cocodataset.org/#overview) dataset to demonstrate how to test 2D
object detection problems on Kolena. Only images with the
[Attribution 2.0](https://creativecommons.org/licenses/by/2.0/) license are included.

## Setup

This project uses [Poetry](https://python-poetry.org/) for packaging and Python dependency management. To get started,
install project dependencies from [`pyproject.toml`](./pyproject.toml) by running:

```shell
poetry update && poetry install
```

## Usage

All data for this example integration lives in the publicly accessible S3 bucket `s3://kolena-public-examples`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.io/installing-kolena/#initialization) for details.

This project defines two scripts that perform the following operations:

1. [`upload_dataset.py`](object_detection_2d/upload_dataset.py) uploads the [COCO](https://cocodataset.org/#overview) dataset - only transportation relevant object annotations are used in this example.

```shell
$ poetry run python3 object_detection_2d/upload_dataset.py --help
usage: upload_dataset.py [-h] [--dataset DATASET]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  Optionally specify a custom dataset name to upload.
```

2. [`upload_results.py`](object_detection_2d/upload_results.py) uploads results for one of the following models: `yolo_r`, `yolo_x`, `mask_rcnn`, `faster_rcnn`, `yolo_v4s`, and `yolo_v3`.

The `upload_results.py` script defines command line arguments to select which model to evaluate — run using the
`--help` flag for more information:

```shell
$ poetry run python3 object_detection_2d/upload_results.py --help
usage: upload_results.py [-h] [--dataset DATASET] {yolo_r,yolo_x,mask_rcnn,faster_rcnn,yolo_v4s,yolo_v3}

positional arguments:
  {yolo_r,yolo_x,mask_rcnn,faster_rcnn,yolo_v4s,yolo_v3}
                        Name of the model to test.

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Optionally specify a custom dataset name to test.
```

## Quality Standards Guide

Once the dataset and results have been uploaded to Kolena, visit [Kolena](https://app.kolena.io/redirect/) to
[explore the data and results](https://docs.kolena.io/dataset/quickstart/#step-3-explore-data-and-results).

Here are our [Quality Standards](https://docs.kolena.io/dataset/core-concepts/quality-standard/) recommendations for object detection:

### Metrics
1. [Precision](https://docs.kolena.io/metrics/precision)
2. [Recall](https://docs.kolena.io/metrics/recall)
3. [F1-score](https://docs.kolena.io/metrics/f1-score)

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
