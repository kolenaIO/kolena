# Example Integration: Age Estimation

This example integration uses the [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/) dataset and
open-source age estimation models to demonstrate how to test regression problems on Kolena.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for packaging and Python dependency management. To get started,
install project dependencies from [`pyproject.toml`](./pyproject.toml) by running:

```shell
uv sync
```

## Usage

The data for this example integration lives in the publicly accessible S3 bucket `s3://kolena-public-examples`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.com/installing-kolena/#initialization) for details.

This project defines two scripts that perform the following operations:

1. [`upload_dataset.py`](age_estimation/upload_dataset.py) creates the Labeled Faces in the Wild dataset on Kolena

```shell
$ uv run age_estimation/upload_dataset.py --help
usage: upload_dataset.py [-h] [--dataset DATASET]

options:
  -h, --help         show this help message and exit
  --dataset DATASET  Optionally specify a custom dataset name to upload.
```

2. [`upload_results.py`](age_estimation/upload_results.py) tests a specified model, e.g. `ssrnet`, on the above dataset

The `upload_results.py` script defines command line arguments to select which model to evaluate — run using the
`--help` flag for more information:

```shell
$ uv run age_estimation/upload_results.py --help
usage: upload_results.py [-h] [--model {ssrnet,deepface}] [--dataset DATASET]

options:
  -h, --help            show this help message and exit
  --model {ssrnet,deepface}
                        Name of model to test.
  --dataset DATASET     Optionally specify a custom dataset name to test.
```
