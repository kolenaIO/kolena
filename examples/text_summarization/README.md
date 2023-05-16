# Example Integration: Text Summarization

This example models the Text Summarization problem with GPT models in Kolena.

## Setup

This project uses [Poetry](https://python-poetry.org/) for packaging and Python dependency management.

Install project dependencies from `examples/text_summarization/pyproject.toml` by running

```zsh
poetry update && poetry install
```

Next, ensure a `KOLENA_TOKEN` environment variable is set. See [initialization instructions](https://docs.kolena.io/testing-with-kolena/using-kolena-client#initialization) for details.

This repository uses pre-commit to run various style and type checks automatically. These same checks are run in CI for all PRs. To set up pre-commit in your local environment, run:

```zsh
poetry run pre-commit install
```

## Running the Text Summarization Workflow

There are two scripts to perform the following operations:

1. [seed_test_suite.py](text_summarization/seed_test_suite.py) creates test suites and test cases
2. [seed_test_run.py](text_summarization/seed_test_run.py) creates model inferences against the test suites

#### Usage

Data lives under the bucket `s3://kolena-public-datasets`.

Make sure there is a set `KOLENA_TOKEN` environment variable. See [initialization instructions](https://docs.kolena.io/testing-with-kolena/using-kolena-client#initialization) for details.

Command line args are defined within each script to specify what model to use and what test suite to seed/evaluate, but a plain end to end run would look like this:

1. Create every test suite and test case:

```zsh
poetry run python3 examples/text_summarization/text_summarization/seed_test_suite.py
```

2. To populate inferences and metrics for a model and test suites, run:

```zsh
poetry run python3 examples/text_summarization/text_summarization/seed_test_run.py --model_name "davinci"
```
