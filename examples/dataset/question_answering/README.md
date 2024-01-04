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

This project defines a script that performs the following operations:

1. [`upload_dataset.py`](question_answering/upload_dataset.py) registers both datasets by default. You can also
select the dataset to register by specifying `--datasets`.
