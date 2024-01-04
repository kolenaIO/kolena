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

The data for this example integration lives in the publicly accessible S3 bucket `s3://kolena-public-datasets`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.io/installing-kolena/#initialization) for details.

This project defines two scripts that perform the following operations:

1. [`upload_dataset.py`](keypoint_detection/upload_dataset.py) creates the 300-W dataset on Kolena

2. [`upload_results.py`](keypoint_detection/upload_results.py) tests a keypoint detection model on the 300-W dataset,
  using either random keypoints (`random` option) or the open-source [RetinaFace](https://github.com/serengil/retinaface)
  (`RetinaFace` option) keypoint detection model.

The `upload_results.py` script defines command line arguments to select which model to evaluate — run using the
`--help` flag for more information:

```shell
$ poetry run python3 keypoint_detection/upload_results.py --help
usage: upload_results.py [-h] {RetinaFace,random} [dataset]

positional arguments:
  {RetinaFace,random}  Name of model to test.
  dataset              Name of dataset to use for testing.

optional arguments:
  -h, --help           show this help message and exit
```
