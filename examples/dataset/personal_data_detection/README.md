# Example Integration: Personal Data Detection

This example uses a subset of the [IMDB dataset](https://developer.imdb.com/non-commercial-datasets/) to demonstrate
checking for the presence of Personal Identifiable Information (PII) using
[the Piiranha model](https://huggingface.co/iiiorg/piiranha-v1-detect-personal-information) prior to uploading the
dataset to Kolena.

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

This project defines a script [`upload_dataset.py`](personal_data_detection/upload_dataset.py) which loads the IMDB
dataset, and runs the Piiranha-v1 model to ensure that there are no PII data before uploading to Kolena. Given that the
Piiranha-v1 model may have false positives, one can optionally specify an allow list of types of PII data to allow in
this check, such as city and username.

```shell
$ uv run personal_data_detection/upload_dataset.py --help
usage: upload_dataset.py [-h] [--dataset DATASET] [--allowed_pii_types ALLOWED_PII_TYPES [ALLOWED_PII_TYPES ...]]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Optionally specify a custom name for the dataset.
  --allowed_pii_types ALLOWED_PII_TYPES [ALLOWED_PII_TYPES ...]
                        Types of PII data to allow in the upload.
                        For instance, to allow surnames and cities, use '--allowed_pii_types I-SURNAME I-CITY'
```
