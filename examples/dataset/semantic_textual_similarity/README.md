# Example Integration: Semantic Textual Similarity

This example integration uses the [STS Benchmark](http://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark) dataset and
open-source sentence transformer models to demonstrate how to test semantic textual similarity problems on Kolena.

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

1. [`upload_dataset.py`](semantic_textual_similarity/upload_dataset.py) uploads the STS Benchmark dataset on Kolena.

2. [`upload_results.py`](semantic_textual_similarity/upload_results.py) tests the following models :
  `distilroberta`, `MiniLM-L12`, `mpnet-base`.

Command line arguments are defined within each script to specify the dataset name to create or model to upload results
for. Run a script using the `--help` flag for more information:

```shell
$ poetry run python3 semantic_textual_similarity/upload_dataset.py --help
usage: upload_dataset.py [-h] [--dataset-csv DATASET_CSV] [--dataset DATASET]

options:
  -h, --help            show this help message and exit
  --dataset-csv DATASET_CSV
                        CSV file specifying dataset. See default CSV for
                        details
  --dataset DATASET     Optionally specify a name of the dataset to upload.

$ poetry run python3 semantic_textual_similarity/upload_results.py --help
usage: upload_results.py [-h] [--dataset DATASET]
                         {distilroberta,MiniLM-L12,mpnet-base}

positional arguments:
  {distilroberta,MiniLM-L12,mpnet-base}
                        Name of the model to test.

options:
  -h, --help            show this help message and exit
  --dataset DATASET     Optionally specify a custom dataset name to test.
```

## Quality Standards Guide

Once the dataset and results have been uploaded to Kolena, visit [Kolena](https://app.kolena.com/redirect/) to
[explore the data and results](https://docs.kolena.com/dataset/quickstart/#step-3-explore-data-and-results).

Here are our [Quality Standards](https://docs.kolena.com/dataset/core-concepts/quality-standard/) recommendations
for semantic textual similarity:

### Metrics

1. Mean Absolute Error (MAE) `mean(abs_error)`
2. Mean Squared Error (MSE) `mean(abs_error_squared)`

### Plots

1. Distribution of `result.error`
2. Distribution of `datapoint.total_char_length`
3. `datapoint.similarity` vs. `result.cos_similarity`

### Test Cases

1. `datapoint.similarity`
2. `datapoint.total_word_count`
3. `datapoint.word_count_diff`
