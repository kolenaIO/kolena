# Example Integration: 2D Object Detection

This example integration uses the [COCO](https://cocodataset.org/#overview) dataset to demonstrate how to test 2D
object detection problems on Kolena.

## Setup

This project uses [Poetry](https://python-poetry.org/) for packaging and Python dependency management. To get started,
install project dependencies from [`pyproject.toml`](./pyproject.toml) by running:

```shell
poetry update && poetry install
```

## Usage

The image data for this example integration lives in the publicly accessible S3 bucket `s3://kolena-public-datasets`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.io/installing-kolena/#initialization) for details.

This project defines two scripts that perform the following operations:

1. [`seed_test_suite.py`](object_detection_2d/seed_test_suite.py) creates the following test suite:

    - `coco-2014-val :: transportation brightness [Object Detection]`, stratified by `light`, `normal`, and `dark`
        brightness

2. [`seed_test_run.py`](object_detection_2d/seed_test_run.py) tests models. #TODO
