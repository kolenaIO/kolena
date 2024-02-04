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

1. [`upload_dataset.py`](text_summarization/upload_dataset.py) uploads the CNN-DailyMail dataset on Kolena.

2. [`upload_results.py`](text_summarization/upload_results.py) tests the following models : `ada`, `babbage`, `curie`,
  `davinci`, `turbo`. Command line arguments are defined to specify what model to test. Run the script using the
  `--help` flag for more information:

Command line arguments are defined within each script to specify the dataset name to create or model to upload results
for. Run a script using the `--help` flag for more information:

```shell
$ poetry run python3 text_summarization/upload_dataset.py --help
usage: upload_dataset.py [-h] [--dataset-csv DATASET_CSV] [--dataset-name DATASET_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --dataset-csv DATASET_CSV
                        CSV file specifying dataset. See default CSV for details
  --dataset DATASET
                        Optionally specify a name of the dataset to upload.

$ poetry run python3 text_summarization/upload_results.py --help
usage: upload_results.py [-h] [--models {ada,babbage,curie,davinci,turbo} [{ada,babbage,curie,davinci,turbo} ...]]
                         [--dataset DATASET]

optional arguments:
  -h, --help            show this help message and exit
  --models {ada,babbage,curie,davinci,turbo} [{ada,babbage,curie,davinci,turbo} ...]
                        Name of model to test.
  --dataset DATASET     Optionally specify a custom dataset name to test.
```

## Quality Standards Guide

Once the dataset and results have been uploaded to Kolena, visit [Kolena](https://app.kolena.io/redirect/) to
[explore the data and results](https://docs.kolena.io/dataset/quickstart/#step-3-explore-data-and-results).

Here are our [Quality Standards](https://docs.kolena.io/dataset/core-concepts/quality-standard/) recommendations for
text summarization:

### Metrics

1. mean(result.BERT_f1)
2. mean(result.ROUGE_1)
3. mean(result.METEOR)
4. mean(result.BLEU)
4. mean(result.cost)

### Plots

1. Distribution of `result.BERT_f1`
2. Distribution of `result.cost`
3. `datapoint.category` vs. `result.BERT_f1`

### Test Cases

1. `datapoint.category`
2. `datapoint.text_word_count`
