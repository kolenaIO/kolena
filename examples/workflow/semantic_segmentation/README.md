# Example Integration: Semantic Segmentation

This example integration uses the [COCO-Stuff 10K](https://github.com/nightrome/cocostuff10k) dataset, specifically
1,789 images with person label, to demonstrate how to test single class semantic segmentation problems on Kolena. Only
images with the [Attribution License](http://creativecommons.org/licenses/by/2.0/),
[Attribution-ShareAlike License](http://creativecommons.org/licenses/by-sa/2.0/),
[Attribution-NoDerivs License](http://creativecommons.org/licenses/by-nd/2.0/) are included.

## Setup

This project uses [Poetry](https://python-poetry.org/) for packaging and Python dependency management. To get started,
install project dependencies from [`pyproject.toml`](./pyproject.toml) by running:

```shell
poetry update && poetry install
```

## Usage

The data for this example integration lives in the publicly accessible S3 bucket `s3://kolena-public-datasets`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.io/installing-kolena/#initialization) for details.

This project defines three scripts that perform the following operations:

1. [`seed_test_suite.py`](semantic_segmentation/seed_test_suite.py) creates the following test suites:

    - `"# of people :: coco-stuff-10k [person]"`, containing samples of COCO-Stuff 10K data, specifically 1,789 images
    with person label, stratified by # of people in the image.

2. [`seed_test_run.py`](semantic_segmentation/seed_test_run.py) tests a specified model,
e.g. `pspnet_r101-d8_4xb4-40k_coco-stuff10k-512x512`, `pspnet_r50-d8_4xb4-20k_coco-stuff10k-512x512`, on the above
test suite.

    As part of the evaluation, result masks (i.e. TP/FP/FN masks) are computed and uploaded to a cloud storage for
    better visualization experience on our webapp. Please use `--out-bucket` argument to provide your AWS S3 bucket
    with write access where these result masks are going to be uploaded to. Also, follow the
    [instructions](https://docs.kolena.io/connecting-cloud-storage/amazon-s3/) to connect
    your bucket to Kolena.

    The result masks will be stored under `s3://{args.out_bucket}/coco-stuff-10k/results/{args.model}` directory in
    your bucket.

3. An optional script, [`seed_activation_map.py`](semantic_segmentation/seed_activation_map.py)
   demonstrates how to generate activation maps from the aformentioned model inferences.
   If you wish to generate your own, use the `--out-bucket` argument to provide an
   AWS S3 bucket where the activation maps will be uploaded to.

Command line arguments are defined within each script to specify what model to use and what test suite to seed/evaluate.
Run a script using the `--help` flag for more information:

```shell
$ poetry run python3 semantic_segmentation/seed_test_run.py --help
usage: seed_test_run.py [-h] [--model MODEL] [--test-suites TEST_SUITES [TEST_SUITES ...]]
                        --out-bucket OUT_BUCKET

options:
  -h, --help            show this help message and exit
  --model MODEL         Name of model in directory to test
  --test-suites TEST_SUITES [TEST_SUITES ...]
                        Name(s) of test suite(s) to test.
  --out-bucket OUT_BUCKET
                        Name of AWS S3 bucket with write access to upload result masks to.
```
