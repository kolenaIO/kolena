# Example Integration: Face Recognition (1:1)

This example integration uses the [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/) dataset and Face Recognition (FR) workflow to demonstrate how to test and evaluate end-to-end FR (1:1) model pipelines on Kolena. The evaluation stages are: face detection (bbox), keypoint extraction, and recognition.

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

1. [`seed_test_suite.py`](face_recognition_11/seed_test_suite.py) creates the following test suites:

    - `labeled-faces-in-the-wild :: gender [FR]`, stratified by `gender` using image metadata
    - `labeled-faces-in-the-wild :: race [FR]`, stratified by `race` using image metadata

    Run this command to seed the default test suite:
    ```shell
    poetry run python3 face_recognition_11/seed_test_suite.py
    ```

    > **NOTE:**  Face bounding box and keypoint ground truths are inferred from [RetinaFace](https://github.com/serengil/retinaface/) as they are not provided in the LFW dataset. Also, for demo purposes, we have subsampled 10,000 pairs, made up of 9,983 images, from LFW.

2. [`seed_test_run.py`](face_recognition_11/seed_test_run.py) tests multiple face recognition models against the test suite above.

    Run this command to evaluate the default models on the `labeled-faces-in-the-wild :: gender [FR]` and `labeled-faces-in-the-wild :: race [FR]` test suites:
    ```shell
    poetry run python3 face_recognition_11/seed_test_run.py
    ```

Command line arguments are defined within each script to specify what model to use and what test suite to seed/evaluate.
Run a script using the `--help` flag for more information:

```shell
$ poetry run python3 face_recognition_11/seed_test_run.py --help
usage: seed_test_run.py [-h] [--models MODELS [MODELS ...]]
                        [--detectors DETECTORS [DETECTORS ...]]
                        [--test_suites TEST_SUITES [TEST_SUITES ...]]

positional arguments:
  models  Name of the model(s) to test.
  test_suites  Name of the test suite(s) to run.

optional arguments:
  -h, --help  show this help message and exit
  --models MODELS [MODELS ...]
                        Name(s) of model(s) in directory to test
  --detectors DETECTORS [DETECTORS ...]
                        Name(s) of detectors(s) used with corresponding model(s).
  --test_suites TEST_SUITES [TEST_SUITES ...]
                        Name(s) of test suite(s) to test.
```
