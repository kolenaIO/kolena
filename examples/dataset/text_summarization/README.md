# Example Integration: Text Summarization

This example integration uses the [CNN-DailyMail](https://paperswithcode.com/dataset/cnn-daily-mail-1) dataset and
OpenAI's [GPT-3](https://platform.openai.com/docs/models/gpt-3) and
[GPT-3.5](https://platform.openai.com/docs/models/gpt-3-5) model families to demonstrate how to test text summarization
problems on Kolena.

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

1. [`upload_dataset.py`](text_summarization/upload_dataset.py) creates CNN-DailyMail dataset on Kolena.

2. [`upload_results.py`](text_summarization/upload_results.py) tests the following models on the above test suites: `ada`,
  `babbage`, `curie`, `davinci`, `turbo`. Command line arguments are defined to specify what model to test. Run the
  script using the `--help` flag for more information:

```shell
$ poetry run python3 text_summarization/upload_results.py --help

```
