# Example Integration: Question Answering

This example integration uses the [Conversational Question Answering (CoQA)](https://stanfordnlp.github.io/coqa/) dataset and OpenAI's GPT models to demonstrate the question answering workflow in Kolena.

## Setup

This project uses [Poetry](https://python-poetry.org/) for packaging and Python dependency management. To get started,
install project dependencies from [`pyproject.toml`](./pyproject.toml) by running:

```shell
poetry update
```

## Usage

The data for this example integration lives in the publicly accessible S3 bucket `s3://kolena-public-datasets`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.io/installing-kolena/#initialization) for details.

This project defines two scripts that perform the following operations:

1. [`seed_test_suite.py`](question_answering/seed_test_suite.py) creates the following test suites:

    - `question types :: CoQA`, stratified into types of questions: `what`, `who`, `how`, `did`, `where`, `was`, `when`, `is`, `why`, and `other`
    - `conversation depths :: CoQA`, stratified by depth of conversation: `1` through `20` interactions

2. [`seed_test_run.py`](question_answering/seed_test_run.py) tests a specified model, e.g. `gpt-4`, on the above test suites

Command line arguments are defined within each script to specify what model to use and what test suite to seed/evaluate.
Run a script using the `--help` flag for more information:

```shell
$ poetry run python3 question_answering/seed_test_run.py --help
usage: seed_test_run.py [-h] [--test_suite TEST_SUITE] [--model {gpt-3.5-turbo-0301,gpt-3.5-turbo,gpt-4-0314,gpt-4}]

optional arguments:
  -h, --help            show this help message and exit
  --test_suite TEST_SUITE
                        Name of the test suite to test.
  --model {gpt-3.5-turbo-0301,gpt-3.5-turbo,gpt-4-0314,gpt-4}
                        Name of the model to test.
```
