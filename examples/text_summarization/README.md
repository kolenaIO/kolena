# Example Integration: Text Summarization

This example integration uses the [CNN-DailyMail](https://paperswithcode.com/dataset/cnn-daily-mail-1) dataset and
OpenAI's [GPT-3](https://platform.openai.com/docs/models/gpt-3) and
[GPT-3.5](https://platform.openai.com/docs/models/gpt-3-5) model families to demonstrate how to test Text Summarization
problems on Kolena.

## Setup

This project uses [Poetry](https://python-poetry.org/) for packaging and Python dependency management.

Install project dependencies from [`pyproject.toml`](./pyproject.toml) by running:

```shell
poetry update && poetry install
```

This repository uses [pre-commit](https://pre-commit.com/) to run various style and type checks automatically. These
same checks are run in CI on all PRs. To set up pre-commit in your local environment, run:

```shell
poetry run pre-commit install
```

## Running the Text Summarization Workflow

The data for this example integration lives in the S3 bucket `s3://kolena-public-datasets`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.io/testing-with-kolena/using-kolena-client#initialization) for
details.

This project defines two scripts that perform the following operations:

1. [seed_test_suite.py](text_summarization/seed_test_suite.py) creates the following test suites:

  - `CNN-DailyMail :: moderation score`, stratified by `very low`, `low`, `medium`, and `high` [moderation scores](https://platform.openai.com/docs/guides/moderation/overview)
  - `CNN-DailyMail :: news category`, stratified by `business`, `entertainment`, `politics`, `tech`, `sport`, and `other`
  - `CNN-DailyMail :: text length`, stratified by `short`, `medium`, and `long` text
  - `CNN-DailyMail :: text X ground truth length`, stratified by the cross product of `short`, `medium`, and `long` text lengths and ground truth lengths

2. [seed_test_run.py](text_summarization/seed_test_run.py) tests the following models on the above test suites: `ada`,
  `babbage`, `curie`, `davinci`, and `turbo`.

Command line arguments are defined within each script to specify what model to use and what test suite to seed/evaluate.
Run a script using the `--help` flag for more information:

```shell
poetry run python3 text_summarization/seed_test_suite --help
```
