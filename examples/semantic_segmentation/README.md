# Example Integration: Semantic Segmentation

This example integration uses the [COCO-Stuff 10K](https://github.com/nightrome/cocostuff10k) dataset to demonstrate how to test single class semantic segmentation problems on Kolena.

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

1. [`seed_test_suite.py`](semantic_segmentation/seed_test_suite.py) creates the following test suites:

    - `"coco-stuff-10k"`, containing samples of COCO-Stuff 10K data

2. [`seed_test_run.py`](semantic_segmentation/seed_test_run.py) tests a specified model, e.g. `pspnet_r101-d8_4xb4-40k_coco-stuff10k-512x512`, on the above test suites

Command line arguments are defined within each script to specify what model to use and what test suite to seed/evaluate.
Run a script using the `--help` flag for more information:

```shell
$ poetry run python3 semantic_segmentation/seed_test_run.py --help
usage: seed_test_run.py [-h] [--test_suite TEST_SUITE] [--models MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --test_suites TEST_SUITE
                        Name(s) of the test suite(s) to test.
  --model MODEL
                        Name(s) of the model(s) to test.
```
