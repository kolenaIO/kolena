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
[initialization documentation](https://docs.kolena.io/installing-kolena/#initialization) for details.

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

### Search Embeddings

> Kolena supports search embedding extraction and upload as an opt-in feature for our customers.
> Please message your point of contact for the latest relevant extractor package.

#### Setup

1. Ensure that the `seed_test_suite.py` script has been successfully run for the `age_estimation` workflow.
2. Copy the `kolena_studio_client_api.tar.gz` file (provided by your Kolena contact) to the `./local_packages` directory.
3. Run `poetry install --extras search`
4. [Recommended] Download test images to a local path for faster embeddings extraction:
   - `mkdir -p local/data/directory/imgs`
   - `aws s3 cp --recursive s3://kolena-public-datasets/labeled-faces-in-the-wild/imgs local/data/directory/imgs`

#### Usage

Ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
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
