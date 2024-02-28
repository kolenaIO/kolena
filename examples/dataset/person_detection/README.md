# Example Integration: Object Detection (2D)

This example integration uses the [COCO](https://cocodataset.org/#overview) dataset to demonstrate how to test single
class 2D object detection problems on Kolena.

## Setup

This project uses [Poetry](https://python-poetry.org/) for packaging and Python dependency management. To get started,
install project dependencies from [`pyproject.toml`](./pyproject.toml) by running:

```shell
poetry update && poetry install
```

## Usage

All data for this example integration lives in the publicly accessible S3 bucket `s3://kolena-public-examples`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.com/installing-kolena/#initialization) for details.

This project defines two scripts that perform the following operations:

1. [`upload_dataset.py`](person_detection/upload_dataset.py) uploads the [COCO](https://cocodataset.org/#overview)
dataset - only person relevant object annotations are used in this example. This includes figures corresponding to
the person keypoints provided with the coco-2014-val annotations.

```shell
$ poetry run python3 person_detection/upload_dataset.py --help
usage: upload_dataset.py [-h] [--dataset DATASET]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  Optionally specify a custom dataset name to upload.
```

2. [`upload_results.py`](person_detection/upload_results.py) uploads results for one of the following
models: `yolo_r`, `yolo_x`, `mask_rcnn`, `faster_rcnn`, `yolo_v4s`, and `yolo_v3`.

The `upload_results.py` script defines command line arguments to select which model to evaluate — run using the
`--help` flag for more information:

```shell
$ poetry run python3 person_detection/upload_results.py --help
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
[explore the data and results](https://docs.kolena.com/dataset/quickstart/#step-3-explore-data-and-results).

Here are our [Quality Standards](https://docs.kolena.com/dataset/core-concepts/quality-standard/) recommendations
for Person Detection:

### Metrics

1. [Precision](https://docs.kolena.com/metrics/precision)
2. [Recall](https://docs.kolena.com/metrics/recall)
3. [F1-score](https://docs.kolena.com/metrics/f1-score)
4. [Average Precision](https://docs.kolena.com/metrics/average-precision)

### Plots

1. `datapoint.calculated_average_bbox_size` vs. `mean(count_FN)`
2. `datapoint.calculated_average_bbox_size` vs. `mean(count_FP)`
3. `datapoint.calculated_average_bbox_size` vs. `mean(count_TP)`
4. `datapoint.calculated_blur_score` vs. `mean(count_FN)`
5. `datapoint.calculated_blur_score` vs. `mean(count_FP)`
6. `datapoint.calculated_blur_score` vs. `mean(count_TP)`
7. `datapoint.calculated_brightness_score` vs. `mean(count_FN)`
8. `datapoint.calculated_brightness_score` vs. `mean(count_FP)`
9. `datapoint.calculated_brightness_score` vs. `mean(count_TP)`

### Test Cases

1. `datapoint.calculated_average_bbox_size`
2. `datapoint.calculated_blur_score`
3. `datapoint.calculated_brightness_score`
