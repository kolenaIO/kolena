# Example Integration: Image Retrieval by Text

This example integration uses the [COCO](https://cocodataset.org/#overview) dataset to demonstrate how to test image
retrieval by text with Kolena. Only images with the [Attribution 2.0](https://creativecommons.org/licenses/by/2.0/)
license are included.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for packaging and Python dependency management. To get started,
install project dependencies from [`pyproject.toml`](./pyproject.toml) by running:

```shell
uv sync
```

Optionally download the COCO images from our s3 bucket to a directory if you want to run inference in real time
instead of uploading a pre-computed results

```shell
mkdir coco_images
aws s3 cp s3://kolena-public-examples/coco-2014-val/data/ coco_images/ --recursive
```

## Usage

All data for this example integration lives in the publicly accessible S3 bucket `s3://kolena-public-examples`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.com/installing-kolena/#initialization) for details.

This project defines two scripts that perform the following operations:

1. [upload_dataset.py](image_retrieval_by_text%2Fupload_dataset.py) uploads the [COCO](https://cocodataset.org/#overview)
dataset.

```shell
$ uv run image_retrieval_by_text/upload_dataset.py --help
usage: upload_dataset.py [-h] [--dataset DATASET]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  Optionally specify a custom dataset name to upload.
```

2. [`upload_results.py`](image_retrieval_by_text/upload_results.py) uploads results for one of the following
models: `kakaobrain/align-base`, `google/siglip-base-patch16-224`, `BAAI/AltCLIP`, and `openai/clip-vit-base-patch32`.

The `upload_results.py` script defines command line arguments to select which model to evaluate — run using the
`--help` flag for more information:

```shell
$ uv run image_retrieval_by_text/upload_results.py --help
usage: upload_results.py [-h] [--model {kakaobrain/align-base,google/siglip-base-patch16-224,BAAI/AltCLIP,openai/clip-vit-base-patch32}]
                         [--dataset DATASET] [--run-inference RUN_INFERENCE] [--local-image-dir LOCAL_IMAGE_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --model {kakaobrain/align-base,google/siglip-base-patch16-224,BAAI/AltCLIP,openai/clip-vit-base-patch32}
                        Name of the model to test. If you want to run inference openai_clip-vit-base-patch32 model
                        is recommended, as it runs faster (should complete in 30 minutes)
  --dataset DATASET     Optionally specify a custom dataset name to test.
  --run-inference RUN_INFERENCE
                        Optionally specify whether to run inference. If this is False, pre-computed inference
                        results will be used
  --local-image-dir LOCAL_IMAGE_DIR
                        Optionally specify a local directory that stores the images to make run inference faster

```

## Quality Standards Guide

Once the dataset and results have been uploaded to Kolena, visit [Kolena](https://app.kolena.com/redirect/) to
[explore the data and results](https://docs.kolena.com/dataset/quickstart/#step-3-explore-data-and-results).

Here are our [Quality Standards](https://docs.kolena.com/dataset/core-concepts/quality-standard/) recommendations
for Image Retrieval By Text:

### Metrics

1. rate(result.is_top_10=true)
2. mean(result.rank)
3. rate(result.total_miss=true)

### Plots

1. Distribution: `datapoint.brightness`
2. Distribution: `datapoint.caption_derived_gender`
3. Distribution: `datapoint.caption_derived_interest`
4. Distribution: `result.is_top_10`
8. `result.is_top_10` vs. `datapoint.caption_derived_interest`
9. `result.is_top_10` vs. `datapoint.caption_derived_gender`

### Test Cases

1. `datapoint.caption_derived_interest`
2. `datapoint.caption_derived_gender`
