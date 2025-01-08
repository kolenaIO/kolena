# Example Integration: Retrieval Augmented Generation (RAG)

This example integration uses the [Financebench](https://github.com/patronus-ai/financebench) dataset to
demonstrate testing RAG system on Kolena.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for packaging and Python dependency management. To get started,
install project dependencies from [`pyproject.toml`](./pyproject.toml) by running:

```shell
uv sync
```

## Usage

The data for this example integration lives in the publicly accessible S3 bucket `s3://kolena-public-examples`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.com/installing-kolena/#initialization) for details.

This project defines three scripts that perform the following operations:

1. [`upload_dataset.py`](retrieval_augmented_generation/upload_dataset.py) creates the Financebench dataset on Kolena

To run it without ground truth, use `s3://kolena-public-examples/financebench/raw/financebench_without_gt.jsonl`
dataset jsonl file instead:

```shell
uv run python retrieval_augmented_generation/upload_dataset.py --dataset-jsonl s3://kolena-public-examples/financebench/raw/financebench_without_gt.jsonl
```

2. [`upload_results.py`](retrieval_augmented_generation/upload_results.py) uploads a RAG system's raw inference
on the Financebench dataset.

The `upload_results.py` script defines command line arguments to select which model to evaluate — run
using the `--help` flag for more information:

```shell
$ uv run python3 retrieval_augmented_generation/upload_results.py --help
usage: upload_results.py [-h] [--dataset-name DATASET_NAME] [--evaluate] [{baseline,qme,query_decomp}]

positional arguments:
  {baseline,qme,query_decomp}
                        Name of the model to test.

optional arguments:
  -h, --help            show this help message and exit
  --dataset-name DATASET_NAME
                        Optionally specify a custom dataset name to test.
  --evaluate            Computes metrics on the model results. Requires dataset with ground truth.
```

3. Label your dataset on [Kolena]((https://app.kolena.com/redirect/))

4. Run evaluation by using `--evaluate` option from the `upload_results.py` script. It will compute metrics on the
model results and upload the model results including the metrics to Kolena.

## Quality Standards Guide

Once the dataset and results have been uploaded to Kolena, visit [Kolena](https://app.kolena.com/redirect/) to
test the rain forecast models. See our [QuickStart](https://docs.kolena.com/dataset/quickstart/) guide
for details.

Here are our [Quality Standards](https://docs.kolena.com/dataset/core-concepts/quality-standard/) recommendations for
this workflow:

### Metrics

1. rate(`result.is_page_retrieved`=true): page-level retrieval rate
2. rate(`result.is_doc_retrieved`=true): doc-level retrieval rate
3. `is_correct` using [LLM prompt](https://docs.kolena.com/dataset/advanced-usage/llm-prompt-extraction/)
