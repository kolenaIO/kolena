# Example Integration: Video Retrieval by Text

This example integration uses the [VATEX](https://eric-xw.github.io/vatex-website/) dataset and
open-source CLIP models to demonstrate how to test video retrieval by text on Kolena.

## Setup

This project uses [Poetry](https://python-poetry.org/) for packaging and Python dependency management. To get started,
install project dependencies from [`pyproject.toml`](./pyproject.toml) by running:

```shell
poetry update && poetry install
```

## Usage

The data for this example integration lives in the publicly accessible S3 bucket `s3://kolena-public-examples`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.com/installing-kolena/#initialization) for details.

This project defines two scripts that perform the following operations:

1. [`upload_dataset.py`](video_retrieval_by_text/upload_dataset.py) uploads the VATEX dataset to Kolena.

```shell
$ poetry run python3 video_retrieval_by_text/upload_dataset.py --help
usage: upload_dataset.py [-h] [-n NAME]

options:
  -h, --help            show this help message and exit
  -n NAME, --name NAME  Optionally specify a custom name for the dataset.
```

2. [`upload_results.py`](video_retrieval_by_text/upload_results.py) tests a specified model, e.g. `CLIP`, on the above dataset.

The `upload_results.py` script defines command line arguments to select which model to evaluate — run using the
`--help` flag for more information:

```shell
$ poetry run python3 video_retrieval_by_text/upload_results.py --help
usage: upload_results.py [-h] [--model {CLIP}] [--dataset DATASET]

options:
  -h, --help            show this help message and exit
  --model {CLIP}
                        Name of model to test.
  --dataset DATASET     Optionally specify a custom dataset name to test.
```
