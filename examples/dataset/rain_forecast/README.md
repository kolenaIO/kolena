# Example Integration: Rain Forecast

This example integration uses the
[Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) tabular dataset to
demonstrate testing rain forecast models on Kolena.

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

1. [`upload_dataset.py`](rain_forecast/upload_dataset.py) creates the Rain in Australia dataset on Kolena

2. [`upload_results.py`](rain_forecast/upload_results.py) tests a rain forecast model on the Rain in Australia dataset.

The `upload_results.py` script defines command line arguments to select which model to evaluate — run using the
`--help` flag for more information:

```shell
$ poetry run python3 rain_forecast/upload_results.py --help
usage: upload_results.py [-h] [--model {ann,logreg}] [--dataset DATASET]

optional arguments:
  -h, --help            show this help message and exit
  --model {ann,logreg}  Name of model to test.
  --dataset DATASET     Name of dataset to use for testing.
```

## Quality Standards Guide

Once the dataset and results have been uploaded to Kolena, visit [Kolena](https://app.kolena.io/redirect/) to
test the rain forecast models. See our [QuickStart](https://docs.kolena.io/dataset/quickstart/) guide
for details.

Here are our [Quality Standards](https://docs.kolena.io/dataset/core-concepts/quality-standard/) recommendations for
this workflow:

### Metrics
1. Precision
2. Recall
3. F1-score
4. \# FP
5. \# FN


### Plots
1. Confusion Matrix: datapoint.RainTomorrow vs. result.will_rain
2. Location vs. mean(result.is_FN)
3. Location vs. mean(result.is_FP)
4. Year vs. mean(result.is_FN)
5. Year vs. mean(result.is_FP)

### Test Cases
1. Year (bin = 5)
2. Region
