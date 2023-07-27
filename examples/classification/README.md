# Example Integration: Classification

This example integration uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and open-source
classification models to demonstrate how to test multiclass classification problems on Kolena.

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

### Multiclass Classification on CIFAR-10

This project defines two scripts that perform the following operations:

1. [`seed_test_suite.py`](scripts/multiclass/seed_test_suite.py) creates the following test suite:

    - `complete cifar10/test [classification]`: complete CIFAR-10

2. [`seed_test_run.py`](scripts/multiclass/seed_test_run.py) tests a specified model, e.g. `resnet50v2`, on the above test suite.

Command line arguments are defined within each script to specify what model to use and what test suite to seed/evaluate.
Run a script using the `--help` flag for more information:

```shell
$ poetry run python3 scripts/multiclass/seed_test_run.py --help
usage: seed_test_run.py [-h] [--models MODELS [MODELS ...]]
                        [--test_suites TEST_SUITES [TEST_SUITES ...]]

optional arguments:
  -h, --help            show this help message and exit
  --models MODELS [MODELS ...]
                        Name(s) of model(s) in directory to test
  --test_suites TEST_SUITES [TEST_SUITES ...]
                        Name(s) of test suite(s) to test.
```
