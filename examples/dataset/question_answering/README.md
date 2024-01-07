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

The data for this example integration lives in the publicly accessible S3 bucket `s3://kolena-public-examples`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.io/installing-kolena/#initialization) for details.

This project defines three scripts that perform the following operations:

1. [`upload_dataset.py`](question_answering/upload_dataset.py) registers both datasets by default. You can also
select the dataset to register by specifying `--datasets`.

```shell
$ poetry run python3 question_answering/upload_dataset.py --help
usage: upload_dataset.py [-h] [--datasets {TruthfulQA,HaluEval-QA} [{TruthfulQA,HaluEval-QA} ...]]

optional arguments:
  -h, --help            show this help message and exit
  --datasets {TruthfulQA,HaluEval-QA} [{TruthfulQA,HaluEval-QA} ...]
                        Name(s) of the dataset(s) to register.
```

2. [optional] [`prepare_results.py`](question_answering/prepare_results.py) computes metrics on each datapoint on both
datasets and saves the results and metrics to a csv file in `s3://kolena-public-examples` bucket. You can also select
the dataset and model to compute metrics by specifying `--datasets` and `--models`. This step is optional as the results
for both datasets and models have already been computed and saved to the cloud storage. NOTE: It will take about 1.5hr
to run the entire set of metrics on a single dataset / model.

```shell
$ poetry run python3 question_answering/prepare_results.py --help
usage: upload_dataset.py [-h] [--datasets {TruthfulQA,HaluEval-QA} [{TruthfulQA,HaluEval-QA} ...]]

optional arguments:
  -h, --help            show this help message and exit
  --datasets {TruthfulQA,HaluEval-QA} [{TruthfulQA,HaluEval-QA} ...]
                        Name(s) of the dataset(s) to register.
```

In order to use the script, make sure the `OPENAI_API_KEY` environment variable is populated in your environment. See
your [OpenAI's User Settings](https://platform.openai.com/api-keys) for the API key.

3. [`upload_results.py`](question_answering/upload_results.py) loads the results csv files from the cloud storage and
uploads pre-computed results (metrics and inferences) for both datasets. You can select the dataset and model to upload
results for by specifying `--datsets` and `--models`.

```shell
$ poetry run python3 question_answering/upload_results.py --help
usage: upload_results.py [-h] [--datasets {TruthfulQA,HaluEval-QA} [{TruthfulQA,HaluEval-QA} ...]]
                         [--models {gpt-3.5-turbo,gpt-4-1106-preview} [{gpt-3.5-turbo,gpt-4-1106-preview} ...]]

optional arguments:
  -h, --help            show this help message and exit
  --datasets {TruthfulQA,HaluEval-QA} [{TruthfulQA,HaluEval-QA} ...]
                        Name(s) of the dataset(s) to test.
  --models {gpt-3.5-turbo,gpt-4-1106-preview} [{gpt-3.5-turbo,gpt-4-1106-preview} ...]
                        Name(s) of the model(s) to test.
```

Now, the datasets and results are uploaded to Kolena. Go to your [datasets page](https://app.kolena.io/redirect/datasets)
to start an investigation on your results by aggregating metrics and building plots to understand your models'
performance.
