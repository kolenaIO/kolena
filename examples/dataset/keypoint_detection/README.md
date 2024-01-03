# Example Integration: Keypoint Detection

This example integration uses the [300 Faces In the Wild (300-W)](https://ibug.doc.ic.ac.uk/resources/300-W/) dataset for a quick run through of keypoint detection testing on Kolena. It closely follows the [quickstart guide](https://docs.kolena.io/building-a-workflow) documentation for building a workflow.

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

1. [`seed_test_suite.py`](keypoint_detection/seed_test_suite.py) creates a test case with all test samples in a test suite

2. [`seed_test_run.py`](keypoint_detection/seed_test_run.py) tests a random keypoint generator against the test suite above

Command line arguments are defined within each script to specify what model to use and what test suite to seed/evaluate.
Run a script using the `--help` flag for more information:

```shell
$ poetry run python3 keypoint_detection/seed_test_suite.py --help
usage: seed_test_suite.py [-h] test_suite

positional arguments:
  test_suite  Name of the test suite to make.

optional arguments:
  -h, --help  show this help message and exit
```

```shell
$ poetry run python3 keypoint_detection/seed_test_run.py --help
usage: seed_test_run.py [-h] model_name test_suite

positional arguments:
  model_name  Name of the model to test.
  test_suite  Name of the test suite to run.

optional arguments:
  -h, --help  show this help message and exit
```
