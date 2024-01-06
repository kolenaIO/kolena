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

The data for this example integration lives in the publicly accessible S3 bucket `s3://kolena-public-examples`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.io/installing-kolena/#initialization) for details.

This project defines two scripts that perform the following operations:

1. [`upload_dataset.py`](age_estimation/upload_dataset.py) creates the Labeled Faces in the Wild dataset on Kolena
2. [`upload_results.py`](age_estimation/upload_results.py) tests a specified model, e.g. `ssrnet`, on the above test suites

The `upload_results.py` script defines command line arguments to select which model to evaluate — run using the
`--help` flag for more information:

```shell
$ poetry run python3 age_estimation/upload_results.py --help
usage: upload_results.py [-h] {ssrnet,deepface}

positional arguments:
  {ssrnet,deepface}  Name of model to test.

options:
  -h, --help         show this help message and exit
```
