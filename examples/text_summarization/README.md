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

The data for this example integration lives in the publicly accessible S3 bucket `s3://kolena-public-datasets`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.io/testing-with-kolena/using-kolena-client#initialization) for
details.

This project defines two scripts that perform the following operations:

1. [`seed_test_suite.py`](text_summarization/seed_test_suite.py) creates the following test suites:

    - `CNN-DailyMail :: moderation score`, stratified by `very low`, `low`, `medium`, and `high`
        [moderation scores](https://platform.openai.com/docs/guides/moderation/overview)
    - `CNN-DailyMail :: news category`, stratified by `business`, `entertainment`, `politics`, `tech`, `sport`, and `other`
    - `CNN-DailyMail :: text length`, stratified by `short`, `medium`, and `long` text
    - `CNN-DailyMail :: text X ground truth length`, stratified by the cross product of `short`, `medium`, and `long`
        text lengths and ground truth lengths

2. [`seed_test_run.py`](text_summarization/seed_test_run.py) tests the following models on the above test suites: `ada`,
  `babbage`, `curie`, `davinci`, `turbo`

Command line arguments are defined within each script to specify what model to use and what test suite to seed/evaluate.
Run a script using the `--help` flag for more information:

```shell
$ poetry run python3 text_summarization/seed_test_run.py --help
usage: seed_test_run.py [-h] [--test-suite TEST_SUITE] [--local-csv LOCAL_CSV] {ada,babbage,curie,davinci,turbo}

positional arguments:
  {ada,babbage,curie,davinci,turbo}
                        The name of the model to test.

optional arguments:
  -h, --help            show this help message and exit
  --test-suite TEST_SUITE
                        Optionally specify a test suite to test. Test against all available test suites when unspecified.
  --local-csv LOCAL_CSV
                        Optionally specify a local results CSV to use. Defaults to CSVs stored in S3 when absent.
```
