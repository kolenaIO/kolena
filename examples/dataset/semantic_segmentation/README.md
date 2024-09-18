# Example Integration: Semantic Segmentation

This example integration uses the [COCO-Stuff 10K](https://github.com/nightrome/cocostuff10k) dataset, specifically
1,789 images with person label, to demonstrate how to test single class semantic segmentation problems on Kolena. Only
images with the [Attribution License](http://creativecommons.org/licenses/by/2.0/),
[Attribution-ShareAlike License](http://creativecommons.org/licenses/by-sa/2.0/),
[Attribution-NoDerivs License](http://creativecommons.org/licenses/by-nd/2.0/) are included.

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

1. [`upload_dataset.py`](semantic_segmentation/upload_dataset.py) creates the COCO-Stuff 10K dataset on Kolena

2. [`upload_results.py`](semantic_segmentation/upload_results.py) tests a specified model,
e.g. `pspnet_r101`, `pspnet_r50`, on the dataset.

    As part of the evaluation, result masks (i.e. TP/FP/FN masks) are computed and uploaded to a cloud storage for
    better visualization experience on our webapp. Please use `--write-bucket` argument to provide your AWS S3 bucket
    with write access where these result masks are going to be uploaded to. Also, follow the
    [instructions](https://docs.kolena.com/connecting-cloud-storage/amazon-s3/) to connect
    your bucket to Kolena.

    The result masks will be stored under `s3://{args.write_bucket}/coco-stuff-10k/results/{args.model}/masks` directory
    in your bucket.

3. An optional script, [`upload_activation_map.py`](semantic_segmentation/upload_activation_map.py) demonstrates how to
generate activation maps from the aformentioned model inferences. If you wish to generate your own, use the
`--write-bucket` argument to provide an AWS S3 bucket where the activation maps will be uploaded to.

Command line arguments are defined within each script to specify what model to use and what dataset and model results to
upload.
Run a script using the `--help` flag for more information:

```shell
$ uv run semantic_segmentation/upload_results.py --help
usage: upload_results.py [-h] --write-bucket WRITE_BUCKET [--dataset DATASET]
                         [{pspnet_r101,pspnet_r50}]

positional arguments:
  {pspnet_r101,pspnet_r50}
                        Name of the model to test.

optional arguments:
  -h, --help            show this help message and exit
  --write-bucket WRITE_BUCKET
                        Name of AWS S3 bucket with write access to upload result masks to.
  --dataset DATASET     Optionally specify a custom dataset name to test.
```

## Quality Standards Guide

Once the dataset and results have been uploaded to Kolena, visit [Kolena](https://app.kolena.com/redirect/) to
test the semantic segmentation models. See our [QuickStart](https://docs.kolena.com/dataset/quickstart/) guide
for details.

Here are our [Quality Standards](https://docs.kolena.com/dataset/core-concepts/quality-standard/) recommendations for
this workflow:

### Metrics

1. [Precision](https://docs.kolena.com/metrics/precision)
2. [Recall](https://docs.kolena.com/metrics/recall)
3. [F1-score](https://docs.kolena.com/metrics/f1-score)
4. [\# FP](https://docs.kolena.com/metrics/tp-fp-fn-tn)
5. [\# FN](https://docs.kolena.com/metrics/tp-fp-fn-tn)

### Plots

1. `mean(results.f1)` vs. `datapoint.image_size`

### Test Cases

1. `datapoint.person_count`
