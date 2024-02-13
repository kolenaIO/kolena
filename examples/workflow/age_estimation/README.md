# Example Integration: Age Estimation

This example integration uses the [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/) dataset and
open-source age estimation models to demonstrate how to test regression problems on Kolena.

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

1. [`seed_test_suite.py`](age_estimation/seed_test_suite.py) creates the following test suites:

    - `age :: labeled-faces-in-the-wild [age estimation]`, stratified into age buckets: `(18, 25]`, `(25, 35]`,
        `(35, 55]`, `(55, 75]`
    - `gender :: labeled-faces-in-the-wild [age estimation]`, stratified by estimated gender
    - `race :: labeled-faces-in-the-wild [age estimation]`, stratified by estimated demographic group

2. [`seed_test_run.py`](age_estimation/seed_test_run.py) tests a specified model, e.g. `ssrnet`, on the above test suites

Command line arguments are defined within each script to specify what model to use and what test suite to seed/evaluate.
Run a script using the `--help` flag for more information:

```shell
$ poetry run python3 age_estimation/seed_test_run.py --help
usage: seed_test_run.py [-h] model test_suites [test_suites ...]

positional arguments:
  model        Name of model in directory to test
  test_suites  Name(s) of test suite(s) to test.

optional arguments:
  -h, --help   show this help message and exit
```
