# Example Integration: Classification

This example integration uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), the
[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) dataset, and open-source classification models to demonstrate
how to test multiclass and binary classification problems on Kolena.

The models used in this example are
[`resnet50v2`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet_v2/ResNet50V2) and
[`inceptionv3`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3/InceptionV3).

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


### Binary Classification

For binary classification, there are two scripts that perform the following operations:

1. [`upload_dataset.py`](classification/binary/upload_dataset.py) registers the
[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) dataset.

```shell
$ poetry run python3 classification/binary/upload_dataset.py --help
usage: upload_dataset.py [-h] [--dataset DATASET]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  Optionally specify a custom dataset name to upload.
```

2. [`upload_results.py`](classification/binary/upload_results.py) uploads results for `"resnet50v2"` or
`"inceptionv3"`.

The `upload_results.py` script defines command line arguments to select which model to evaluate — run using the
`--help` flag for more information:

```shell
$ poetry run python3 classification/binary/upload_results.py --help
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

1. [`upload_dataset.py`](classification/multiclass/upload_dataset.py) registers the
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

```shell
$ poetry run python3 classification/multiclass/upload_dataset.py --help
usage: upload_dataset.py [-h] [--dataset DATASET]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  Optionally specify a custom dataset name to upload.
```

2. [`upload_results.py`](classification/multiclass/upload_results.py) uploads results for `"resnet50v2"` or
`"inceptionv3"`.

The `upload_results.py` script defines command line arguments to select which model to evaluate — run using the
`--help` flag for more information:

```shell
$ poetry run python3 classification/multiclass/upload_results.py --help
usage: upload_results.py [-h] [--dataset DATASET] {resnet50v2,inceptionv3}

positional arguments:
  {resnet50v2,inceptionv3}
                        Name of the model to test.

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Optionally specify a custom dataset name to test.
```
