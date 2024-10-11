# Example Extraction: Image Search Embeddings

> Kolena supports search embedding extraction and upload as an opt-in feature for our customers.

This example runs against data from the [`semantic_segmentation`](../semantic_segmentation) example dataset, and assumes
this data has already been uploaded to your Kolena platform.

## Setup

1. Ensure that data for the [`semantic_segmentation`](../semantic_segmentation) dataset has been seeded through calling
the [`upload_dataset.py`](../semantic_segmentation/semantic_segmentation/upload_dataset.py) script.
2. Install the kolena-embeddings package via the natural language search [instructions](https://docs.kolena.com/dataset/advanced-usage/set-up-natural-language-search/#uv)
3. This project uses [uv](https://docs.astral.sh/uv/) for packaging and Python dependency management. Install project
dependencies from [`pyproject.toml`](./pyproject.toml) by running:

```shell
uv sync
```

4. [Recommended] Download test images to a local path for faster embedding extraction:

```shell
mkdir -p local/data/directory/imgs
aws s3 cp --recursive s3://kolena-public-examples/coco-stuff-10k/data/images/ local/data/directory/imgs
```

## Usage

The data for this example integration lives in the publicly accessible S3 bucket `s3://kolena-public-datasets`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.com/installing-kolena/#initialization) for details.

[`upload_embeddings.py`](search_embeddings/upload_embeddings.py) loads data from a publicly accessible S3 bucket, extracts
embeddings or uses pre-extracted embeddings, and uploads them to the Kolena platform.

```shell
$ uv run search_embeddings/upload_embeddings.py --help
usage: upload_embeddings.py [-h] [--run-extraction {True,False}] [--dataset-name DATASET_NAME] [--local-path LOCAL_PATH]

options:
  -h, --help            show this help message and exit
  --run-extraction {True,False}
                        Whether to run extraction. A set of pre-extracted embeddings will be used if set to False.
  --dataset-name DATASET_NAME
                        Optionally specify a name of the dataset to upload embeddings
  --local-path LOCAL_PATH
                        Local path where files have already been pre-downloaded
                        (to the same relative path as s3://kolena-public-examples/coco-stuff-10k/data/images/)
```

Once the embeddings have been extracted, you will be able to search the relevant
age estimation test suite in the [Kolena Studio](https://app.kolena.com/redirect/studio) using natural language.
