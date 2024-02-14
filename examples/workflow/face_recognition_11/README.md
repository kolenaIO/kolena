# Example Integration: Face Recognition (1:1)

This example integration uses the [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/)
dataset and Face Recognition (FR) workflow to
demonstrate how to test and evaluate end-to-end FR (1:1) model pipelines on Kolena.
The evaluation stages are: face detection (bbox), keypoint extraction, and recognition.

## Setup

This project uses [Poetry](https://python-poetry.org/) for packaging and Python dependency management. To get started,
install project dependencies from [`pyproject.toml`](./pyproject.toml) by running:

```shell
poetry update && poetry install
```

## Usage

The data for this example integration lives in the publicly accessible S3 bucket `s3://kolena-public-datasets`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.com/installing-kolena/#initialization) for details.

This project defines two scripts that perform the following operations:

1. [`seed_test_suite.py`](face_recognition_11/seed_test_suite.py) creates the following test suites:

    - `labeled-faces-in-the-wild :: gender [FR]`, stratified by `gender` using image metadata
    - `labeled-faces-in-the-wild :: race [FR]`, stratified by `race` using image metadata

    Run this command to seed the default test suite:

    ```shell
    poetry run python3 face_recognition_11/seed_test_suite.py
    ```

    > **NOTE:**  Face bounding box and keypoint ground truths are inferred
    > from [RetinaFace](https://github.com/serengil/retinaface/) as they are not provided in the LFW dataset.
    > Also, for demo purposes, we have subsampled 29,400 pairs, made up of 9,947 images, from LFW.

2. [`seed_test_run.py`](face_recognition_11/seed_test_run.py)
   tests multiple face recognition models (i.e. vgg-face, facenet512) against the test suite above.

    Run this command to evaluate the default models on
    the `labeled-faces-in-the-wild :: gender [FR]` and `labeled-faces-in-the-wild :: race [FR]` test suites:

    ```shell
    poetry run python3 face_recognition_11/seed_test_run.py
    ```

Command line arguments are defined within each script to specify what model to use and what test suite to seed/evaluate.
Run a script using the `--help` flag for more information:

```shell
$ poetry run python3 face_recognition_11/seed_test_run.py --help
usage: seed_test_run.py [-h] [--models MODELS] [--detectors DETECTORS] [--test-suites TEST_SUITES]

options:
  -h, --help            show this help message and exit
  --models MODELS       Name(s) of model(s) in directory to test
  --detectors DETECTORS
                        Name(s) of detectors(s) used with corresponding model(s).
  --test-suites TEST_SUITES
                        Name(s) of test suite(s) to test.
```
