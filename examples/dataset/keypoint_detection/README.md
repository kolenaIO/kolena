# Example Integration: Keypoint Detection

This example integration uses the [300 Faces In the Wild (300-W)](https://ibug.doc.ic.ac.uk/resources/300-W/) dataset
to demonstrate testing keypoint detection models on Kolena.

## Setup

This project uses [Poetry](https://python-poetry.org/) for packaging and Python dependency management. To get started,
install project dependencies from [`pyproject.toml`](./pyproject.toml) by running:

```shell
poetry update && poetry install
```

## Usage

The data for this example integration lives in the publicly accessible S3 bucket `s3://kolena-public-examples`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.com/installing-kolena/#initialization) for details.

This project defines two scripts that perform the following operations:

1. [`upload_dataset.py`](keypoint_detection/upload_dataset.py) uploads the 300-W dataset.

2. [`upload_results.py`](keypoint_detection/upload_results.py) tests a keypoint detection model on the 300-W dataset,
  using either random keypoints (`random` option) or the open-source [RetinaFace](https://github.com/serengil/retinaface)
  (`retinaface` option) keypoint detection model.

Command line arguments are defined within each script to specify the dataset name to create or model to upload results
for. Run a script using the `--help` flag for more information:

```shell
$ poetry run python3 keypoint_detection/upload_dataset.py --help
usage: upload_dataset.py [-h] [--dataset DATASET]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  Optionally specify a custom dataset name to upload.

$ poetry run python3 keypoint_detection/upload_results.py --help
usage: upload_results.py [-h] [--dataset DATASET] {retinaface,random}

positional arguments:
  {retinaface,random}  Name of the model to test.

optional arguments:
  -h, --help           show this help message and exit
  --dataset DATASET    Optionally specify a custom dataset name to test.
```

## Quality Standards Guide

Once the dataset and results have been uploaded to Kolena, visit [Kolena](https://app.kolena.com/redirect/) to
[explore the data and results](https://docs.kolena.com/dataset/quickstart/#step-3-explore-data-and-results).

Here are our [Quality Standards](https://docs.kolena.com/dataset/core-concepts/quality-standard/) recommendations for
keypoint detection:

### Metrics

1. mean(result.mse)
2. mean(results.nsme)

### Plots

1. `datapoint.normalization_factor` vs. `mean(result.mse)`
2. `datapoint.condition` vs. `mean(result.nmse)`

### Test Cases

1. `datapoint.condition`
2. `datapoint.normalization_factor`
