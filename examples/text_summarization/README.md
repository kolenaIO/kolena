# Example Integration: Text Summarization

This example models the Text Summarization problem with GPT models in Kolena.

## Setup

This project uses [Poetry](https://python-poetry.org/) for packaging and Python dependency management.

Install project dependencies from `examples/text_summarization/pyproject.toml` by running:

```zsh
poetry update && poetry install
```

This repository uses pre-commit to run various style and type checks automatically. These same checks are run in CI for all PRs. To set up pre-commit in your local environment, run:

```zsh
poetry run pre-commit install
```

## Running the Text Summarization Workflow

Data lives in the bucket `s3://kolena-public-datasets`.

Make sure there is a set `KOLENA_TOKEN` environment variable. See [initialization instructions](https://docs.kolena.io/testing-with-kolena/using-kolena-client#initialization) for details.

There are two scripts to perform the following operations:

1. [seed_test_suite.py](text_summarization/seed_test_suite.py) creates test suites and test cases

2. [seed_test_run.py](text_summarization/seed_test_run.py) creates model inferences against the test suites

Created test suites:
 - `CNN-DailyMail :: moderation score`, stratified by `very low`, `low`, `medium`, and `high` [moderation scores](https://platform.openai.com/docs/guides/moderation/overview)
 - `CNN-DailyMail :: news category`, stratified by `business`, `entertainment`, `politics`, `tech`, `sport`, and `other`
 - `CNN-DailyMail :: text length`, stratified by `short`, `medium`, and `long` text
 - `CNN-DailyMail :: text X ground truth length`, stratified by the cross product of `short`, `medium`, and `long` text lengths and ground truth lengths

Available models: `ada`, `babbage`, `curie`, `davinci`, and `turbo`

Command line args are defined within each script to specify what model to use and what test suite to seed/evaluate, but a plain end to end run would look like this:

1. Create every test suite and test case:

```zsh
poetry run python3 examples/text_summarization/text_summarization/seed_test_suite.py
```

2. To populate inferences and metrics for a model and test suites, run:

```zsh
poetry run python3 examples/text_summarization/text_summarization/seed_test_run.py --model_name "davinci"
```
