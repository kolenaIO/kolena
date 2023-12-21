# Example Integration: Question Answering

This example integration uses the [TruthfulQA (open-domain)](https://github.com/sylinrl/TruthfulQA) and the
[HaluEval (closed-domain)](https://github.com/RUCAIBox/HaluEval/tree/main/evaluation) datasets and OpenAI's GPT models
to demonstrate the question answering workflow in Kolena.

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

This project defines three scripts that perform the following operations:

1. [`register_dataset.py`](question_answering/register_dataset.py) registers both datasets by default. You can also
select the dataset to register by specifying `--datasets`.

2. [optional] [`prepare_results.py`](question_answering/prepare_results.py) computes metrics on each datapoint on both
datasets and saves the results and metrics to a csv file in `s3://kolena-public-datasets` bucket. You can also select
the dataset and model to compute metrics by specifying `--datasets` and `--models`. This step is optional as the results
are already computed and saved in the cloud storage.
