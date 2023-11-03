# Example Extraction: Image Search Embeddings

> Kolena supports search embedding extraction and upload as an opt-in feature for our customers.
> Please message your point of contact for the latest relevant extractor package.

This example runs against data from the [`age_estimation`](../age_estimation) example workflow, and assumes this
data has already been uploaded to your Kolena platform.

## Setup

1. Ensure that data for the [`age_estimation`](../age_estimation) workflow has been seeded through calling the
[`seed_test_suite.py`](../age_estimation/age_estimation/seed_test_suite.py) script.
2. Copy the `kolena_embeddings-*.*.*.tar.gz` file (provided by your Kolena contact) to the `./local_packages` directory to a file named `kolena_embeddings.tar.gz`.
3. This project uses [Poetry](https://python-poetry.org/) for packaging and Python dependency management. Install project
dependencies from [`pyproject.toml`](./pyproject.toml) by running:

```shell
poetry update && poetry install
```

4. [Recommended] Download test images to a local path for faster embeddings extraction:
```shell
mkdir -p local/data/directory/imgs
aws s3 cp --recursive s3://kolena-public-datasets/labeled-faces-in-the-wild/imgs local/data/directory/imgs
```

## Usage

The data for this example integration lives in the publicly accessible S3 bucket `s3://kolena-public-datasets`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.io/installing-kolena/#initialization) for details.

[`extract_embeddings.py`](search/extract_embeddings.py) loads data from a publicly accessible S3 bucket, extracts embeddings,
and uploads them to the Kolena platform.

```shell
$ poetry run python3 search/extract_embeddings.py --help
usage: extract_embeddings.py [-h] [--local-path LOCAL_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --local-path LOCAL_PATH
                        Local path where files have already been pre-downloaded (to the same relative path as s3://kolena-public-datasets/labeled-faces-in-the-wild/imgs/)
```

Once the embeddings have been extracted, you will be able to search the relevant age estimation test suite in the [Kolena Studio](https://app.kolena.io/redirect/studio) using natural language.
