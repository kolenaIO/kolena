# Example Integration: 2D Object Detection

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

All data for this example integration lives in the publicly accessible S3 bucket `s3://kolena-public-datasets`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.io/installing-kolena/#initialization) for details.

This project defines two scripts that perform the following operations:

1. [`seed_test_suite.py`](object_detection_2d/seed_test_suite.py) creates the following test suite:

    - `coco-2014-val :: transportation brightness [Object Detection]`, stratified by `light`, `normal`, and `dark`
        brightness

2. [`seed_test_run.py`](object_detection_2d/seed_test_run.py) tests the following models on the above test suites:
  `yolo_r`, `yolo_x`, `mask_cnn`, `faster_rcnn`, `yolo_v4s`, and `yolo_v3`. Information about these models can be
  found in [`constants.py`](object_detection_2d/constants.py).

Command line arguments are defined within each script to specify what model to use and what test suite to
  seed/evaluate. Run a script using the `--help` flag for more information:

```shell
$ poetry run python3 object_detection_2d/seed_test_run.py --help
usage: seed_test_run.py [-h] model {yolo_r,yolo_x,mask_cnn,faster_rcnn,yolo_v4s,yolo_v3}
  [--test_suite TEST_SUITE]

positional arguments:
  {yolo_r,yolo_x,mask_cnn,faster_rcnn,yolo_v4s,yolo_v3}
                        The alias of the model to test.

optional arguments:
  -h, --help            show this help message and exit
  --test-suite TEST_SUITE
                        Optionally specify a test suite to test. Test against all available test suites when unspecified.
```
