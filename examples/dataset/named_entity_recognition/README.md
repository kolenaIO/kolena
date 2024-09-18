# Example Integration: Named Entity Recognition

This example integration uses the [n2c2 2014: National NLP Clinical Challenges](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/)
dataset to demonstrate testing named entity recognition models on Kolena.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for packaging and Python dependency management. To get started,
install project dependencies from [`pyproject.toml`](./pyproject.toml) by running:

```shell
# Include if using torch on cpu
export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu/"
uv sync
```

#### Data

The scripts assume the existence a `testing-PHI-Gold-fixed.tar.gz` file in the `data/` directory. Due to redistribution
limitations, this file is not included with example code, but the user may access and retrieve this data directly
from the [n2c2 NLP Research Data Portal](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/).

## Usage

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.com/installing-kolena/#initialization) for details.

This project defines two scripts that perform the following operations:

1. [`upload_dataset.py`](named_entity_recognition/upload_dataset.py) uploads the n2c2-2014 dataset.

2. [`upload_results.py`](named_entity_recognition/upload_results.py) tests a named entity recognition model on
  the n2c2-2014 dataset, using either a [RoBERTa model](https://huggingface.co/obi/deid_roberta_i2b2) or
  a [BERT model](https://huggingface.co/obi/deid_bert_i2b2) fine-tuned for deidentification of medical notes.

Command line arguments are defined within each script to specify the dataset name to create or model to upload results
for. Run a script using the `--help` flag for more information:

```shell
$ uv run named_entity_recognition/upload_dataset.py --help
usage: upload_dataset.py [-h] [--dataset DATASET]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  Optionally specify a custom dataset name to upload.

$ uv run named_entity_recognition/upload_results.py --help
usage: upload_results.py [-h] [--dataset DATASET] {bert,roberta}

positional arguments:
  {bert,roberta}  Name of the model to test.

optional arguments:
  -h, --help           show this help message and exit
  --dataset DATASET    Optionally specify a custom dataset name to test.
```

## Quality Standards Guide

Once the dataset and results have been uploaded to Kolena, visit [Kolena](https://app.kolena.com/redirect/) to
[explore the data and results](https://docs.kolena.com/dataset/quickstart/#step-3-explore-data-and-results).

Here are our [Quality Standards](https://docs.kolena.com/dataset/core-concepts/quality-standard/) recommendations for
keypoint detection:

### Metrics

These metrics can be added via the "Object Detection" type Task Metric.

1. Precision (Macro)
2. Recall (Macro)
3. F1 Score (Macro)

### Plots

These plots depend on fields hydrated by performing automated text extractions on the `datapoint.text` field.

1. `datapoint.text.word_count` vs. `mean(result.counts.LOC_ERROR)`
2. `datapoint.text.word_count` vs. `mean(result.counts.CLS_ERROR)`

### Test Cases

These test cases depend on fields hydrated by performing automated text extractions on the `datapoint.text` field.

1. `datapoint.text.sentence_count`
2. `datapoint.text.named_entity_count`
