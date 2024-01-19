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
usage: upload_results.py [-h] [--dataset DATASET] {ann,logreg}

positional arguments:
  {ann,logreg}       Name of the model to test.

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  Optionally specify a custom dataset name to test.
```

## Quality Standards Guide

Once the dataset and results have been uploaded to Kolena, visit [Kolena](https://app.kolena.io/redirect/) to
test the rain forecast models. See our [QuickStart](https://docs.kolena.io/dataset/quickstart/) guide
for details.

Here are our [Quality Standards](https://docs.kolena.io/dataset/core-concepts/quality-standard/) recommendations for
this workflow:

### Metrics
1. [Precision](https://docs.kolena.io/metrics/precision)
2. [Recall](https://docs.kolena.io/metrics/recall)
3. [F1-score](https://docs.kolena.io/metrics/f1-score)
4. [\# FP](https://docs.kolena.io/metrics/tp-fp-fn-tn)
5. [\# FN](https://docs.kolena.io/metrics/tp-fp-fn-tn)

### Plots
1. Confusion Matrix: `datapoint.RainTomorrow` vs. `result.will_rain`
2. `datapoint.Location` vs. `mean(result.is_FN)`
3. `datapoint.Location` vs. `mean(result.is_FP)`
4. `datapoint.Year/Month` vs. `mean(result.is_FN)`
5. `datapoint.Year/Month` vs. `mean(result.is_FP)`

### Test Cases
1. `datapoint.Year` (with `bin` = 5)
2. `datapoint.Region`
