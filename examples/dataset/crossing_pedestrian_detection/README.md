# Example Integration: Crossing Pedestrian Detection

This example integration uses the video data from the JAAD dataset to
demonstrate how video data is represented on Kolena.

## Dataset Information

[![](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-1609.04741-b31b1b.svg)](https://arxiv.org/abs/1609.04741)
[![website](https://img.shields.io/badge/website-JAAD-green.svg)](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/)

[**Joint Attention in Autonomous Driving** (JAAD)](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/)
is an open-source dataset designed for **object detection** and **action classification** in videos.
It concentrates on addressing the critical challenge of identifying high-risk pedestrians—those not detected as
crossing—utilizing a range of video data sourced from real-world driving scenarios.
The objective is to enhance the safety of autonomous driving systems by improving the accuracy of pedestrian detection,
especially in scenarios with hazardous potential.

#### Model Details

We use some state-of-the-art object tracking algorithms involving
[Simple Online and Realtime Tracking (SORT)](https://arxiv.org/abs/1602.00763),
including: [`c3d_deepsort`](https://vlg.cs.dartmouth.edu/c3d/),
[`c3d_sort`](https://vlg.cs.dartmouth.edu/c3d/), [`static_sort`](https://arxiv.org/abs/1602.00763),
and [`static_deepsort`](https://arxiv.org/abs/1703.07402).

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for packaging and Python dependency management. To get started,
install project dependencies from [`pyproject.toml`](./pyproject.toml) by running:

```shell
uv sync
```

## Usage

All data for this example integration lives in the publicly accessible S3 bucket `s3://kolena-public-examples`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.com/installing-kolena/#initialization) for details.

This project defines two scripts that perform the following operations:

1. [`upload_dataset.py`](crossing_pedestrian_detection/upload_dataset.py) uploads
   the [JAAD]((https://data.nvision2.eecs.yorku.ca/JAAD_dataset/) ) dataset.

```shell
$ uv run crossing_pedestrian_detection/upload_dataset.py --help
usage: upload_dataset.py [-h] [--dataset DATASET]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  Optionally specify a custom dataset name to upload.
```

2. [`upload_results.py`](crossing_pedestrian_detection/upload_results.py) uploads results for one of the following
   models: `c3d_sort`, `c3d_deepsort`, `static_sort` and `static_deepsort`

```shell
$ uv run crossing_pedestrian_detection/upload_results.py --help
usage: upload_results.py [-h] [--dataset DATASET] {c3d_sort,c3d_deepsort,static_sort,static_deepsort}

positional arguments:
  {c3d_sort,c3d_deepsort,static_sort,static_deepsort}
                        Name of the model to test.

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Optionally specify a custom dataset name to test.
```
