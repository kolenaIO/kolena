# Example Integration: Classification

This example integration uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), the
[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) dataset, and open-source classification models to demonstrate
how to test multiclass and binary classification problems on Kolena.

The models used in this example are
[`resnet50v2`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet_v2/ResNet50V2) and
[`inceptionv3`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3/InceptionV3).

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

### Binary Classification

For binary classification, there are two scripts that perform the following operations:

1. [`upload_dataset.py`](classification/binary/upload_dataset.py) uploads the
[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) dataset.

2. [`upload_results.py`](classification/binary/upload_results.py) uploads results for `"resnet50v2"` or
`"inceptionv3"`.

Command line arguments are defined within each script to specify the dataset name to create or model to upload results
for. Run a script using the `--help` flag for more information:

```shell
$ uv run classification/binary/upload_dataset.py --help
usage: upload_dataset.py [-h] [--dataset DATASET]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  Optionally specify a custom dataset name to upload.

$ uv run classification/binary/upload_results.py --help
usage: upload_results.py [-h] [--dataset DATASET] {resnet50v2,inceptionv3}

positional arguments:
  {resnet50v2,inceptionv3}
                        Name of the model to test.

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Optionally specify a custom dataset name to test.
```

### Multiclass Classification

For multiclass classification, there are two scripts that perform the following operations:

1. [`upload_dataset.py`](classification/multiclass/upload_dataset.py) uploads the
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

2. [`upload_results.py`](classification/multiclass/upload_results.py) uploads results for `"resnet50v2"` or
`"inceptionv3"`.

Command line arguments are defined within each script to specify the dataset name to create or model to upload results
for. Run a script using the `--help` flag for more information:

```shell
$ uv run classification/multiclass/upload_dataset.py --help
usage: upload_dataset.py [-h] [--dataset DATASET]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  Optionally specify a custom dataset name to upload.

$ uv run classification/multiclass/upload_results.py --help
usage: upload_results.py [-h] [--dataset DATASET] {resnet50v2,inceptionv3}

positional arguments:
  {resnet50v2,inceptionv3}
                        Name of the model to test.

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Optionally specify a custom dataset name to test.
```

## Quality Standards Guide

Once the dataset and results have been uploaded to Kolena, visit [Kolena](https://app.kolena.com/redirect/) to
[explore the data and results](https://docs.kolena.com/dataset/quickstart/#step-3-explore-data-and-results).

Here are our [Quality Standards](https://docs.kolena.com/dataset/core-concepts/quality-standard/) recommendations for
classification:

### Metrics

1. [Precision](https://docs.kolena.com/metrics/precision)
2. [Recall](https://docs.kolena.com/metrics/recall)
3. [F1-score](https://docs.kolena.com/metrics/f1-score)
4. [Accuracy](https://docs.kolena.com/metrics/accuracy)

### Binary Classification Plots

1. Confusion Matrix: `datapoint.label.label` vs. `result.inference.label`
2. `result.inference.label` vs. `result.inference.score`

### Multiclass Classification Plots

1. Confusion Matrix: `datapoint.ground_truth.label` vs. `result.classification.label`
2. `result.classification.label` vs. `result.classification.score`

### Test Cases

1. `datapoint.label.label` (`datapoint.ground_truth.label` for multiclass)
