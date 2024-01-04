# Example Integration: Classification

This example integration uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), the
[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) and open-source classification models to demonstrate how to test
multiclass and binary classification problems on Kolena.

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


### Binary Classification

For binary classification, there are two scripts that perform the following operations:

1. [`register_dataset.py`](classification/binary/register_dataset.py) registers the
[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) dataset.

2. [`upload_results.py`](classification/binary/upload_results.py) uploads results for `"resnet50v2"` and
`"inceptionv3"` by default. You can also choose one model by specifying `--datasets`.

The `upload_results.py` script defines command line arguments to select which model to evaluate — run using the
`--help` flag for more information:

```shell
$ poetry run python3 classification/binary/upload_results.py --help
usage: upload_results.py [-h] [--models MODELS [MODELS ...]] [--multiclass]

optional arguments:
  -h, --help            show this help message and exit
  --models MODELS [MODELS ...]
                        Model(s) from ['resnet50v2', 'inceptionv3'] to test
  --multiclass          Option to evaluate dogs-vs-cats as multiclass classification
```


<!---
TODO, indicating where multiclass information will be.
### Multiclass Classification

For multiclass classification, there are two scripts that perform the following operations:

1. [`register_dataset.py`](classification/multiclass/register_dataset.py) registers the
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

2. [`upload_results.py`](classification/multiclass/upload_results.py) uploads results for `"resnet50v2"` and
`"inceptionv3"` by default. You can also choose one model by specifying `--datasets`.

-->
