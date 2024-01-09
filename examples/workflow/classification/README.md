# Example Integration: Classification

This example integration uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), the
[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) and open-source
classification models to demonstrate how to test multiclass and binary classification problems on Kolena.

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

    - `image properties :: cifar10/test`, stratified by `image_brightness` and `image_contrast`

    Run this command to seed the default test suite:
    ```shell
    poetry run python3 scripts/multiclass/seed_test_suite.py
    ```


2. [`seed_test_run.py`](scripts/multiclass/seed_test_run.py) tests a specified model, e.g. `resnet50v2`, on the above test suite.

    Run this command to evaluate the default models on `image properties :: cifar10/test` test suite:
    ```shell
    poetry run python3 scripts/multiclass/seed_test_run.py
    ```

Command line arguments are defined within each script to specify what model to use and what test suite to seed/evaluate.
Run a script using the `--help` flag for more information:

```shell
$ poetry run python3 scripts/multiclass/seed_test_run.py --help
usage: seed_test_run.py [-h] [--models MODELS [MODELS ...]]
                        [--test-suites TEST_SUITES [TEST_SUITES ...]]

optional arguments:
  -h, --help            show this help message and exit
  --models MODELS [MODELS ...]
                        Name(s) of model(s) in directory to test
  --test-suites TEST_SUITES [TEST_SUITES ...]
                        Name(s) of test suite(s) to test.
```


### Binary Classification on Dogs vs. Cats

This project defines two scripts that perform the following operations:

1. [`seed_test_suite.py`](scripts/binary/seed_test_suite.py) creates the following test suite:

    - `image size :: dogs-vs-cats`, stratified by `image size` using metadata `width` and `height`

    Run this command to seed the default test suite:
    ```shell
    poetry run python3 scripts/binary/seed_test_suite.py
    ```

2. [`seed_test_run.py`](scripts/binary/seed_test_run.py) tests a specified model, e.g. `resnet50v2`, on the above test suite.

    Run this command to evaluate the default models on `image size :: dogs-vs-cats` test suite:
    ```shell
    poetry run python3 scripts/binary/seed_test_run.py
    ```


Command line arguments are defined within each script to specify what model to use and what test suite to seed/evaluate.
Run a script using the `--help` flag for more information:

```shell
$ poetry run python3 scripts/binary/seed_test_run.py --help
usage: seed_test_run.py [-h] [--models MODELS [MODELS ...]]
                        [--test-suites TEST_SUITES [TEST_SUITES ...]]
                        [--multiclass]

optional arguments:
  -h, --help            show this help message and exit
  --models MODELS [MODELS ...]
                        Name(s) of model(s) in directory to test
  --test-suites TEST_SUITES [TEST_SUITES ...]
                        Name(s) of test suite(s) to test.
  --multiclass          Option to evaluate dogs-vs-cats as multiclass
                        classification
```
