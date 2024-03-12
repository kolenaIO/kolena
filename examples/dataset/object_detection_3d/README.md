# Example Integration: Object Detection (3D)

This example integration uses the image and LIDAR data from the KITTI dataset to
demonstrate how to test 3D Object Detection on Kolena.

## Dataset Information

![License: CC BY-NC-SA 3.0](https://licensebuttons.net/l/by-nc-sa/3.0/80x15.png)
[![IEEE](https://img.shields.io/badge/IEEE-6248074-b31b1b.svg)](https://ieeexplore.ieee.org/document/6248074)
[![dataset](https://img.shields.io/badge/dataset-KITTI-green.svg)](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
[![website](https://img.shields.io/badge/website-KITTI-green.svg)](https://www.cvlibs.net/datasets/kitti/index.php)

The [KITTI](https://www.cvlibs.net/datasets/kitti/index.php) dataset is a foundational benchmark for **3D Object
Detection** in the field of autonomous driving. It provides a comprehensive suite of high-resolution stereo images and
corresponding 3D point clouds.

The objective for 3D object detection models is to identify relevant entities within a scene—such as vehicles,
pedestrians, and cyclists—but also to precisely determine their sizes, orientations, and positions relative to the
sensor. This task is pivotal in ensuring the safety and reliability of autonomous driving technologies by enabling
vehicles to make informed decisions based on a detailed understanding of their immediate environment.

| Feature                                                                                      | Demo                                                                                                                         |
|----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| Explore 3D point clouds in the Studio by clicking the `velodyne` point cloud asset (`*.pcd`) | ![Explore 3D point clouds in the Studio](https://kolena-public-assets.s3.us-west-2.amazonaws.com/gifs/kitti-point-cloud.gif) |

#### Custom Metadata

Kolena hydrates each video with additional metadata fields with counts of pedestrians, cyclists, and cars.

#### Model Details

For this example, we use [PointPillars](https://arxiv.org/abs/1812.05784)
and [Part-A2](https://arxiv.org/abs/1907.03670) models on three different
F1-optimal configurations for difficulty: `easy`,`moderate`, and `hard`.

## Setup

This project uses [Poetry](https://python-poetry.org/) for packaging and Python dependency management. To get started,
install project dependencies from [`pyproject.toml`](./pyproject.toml) by running:

```shell
poetry update && poetry install
```

This example integration will use pre-trained models provided by
[MMDetection3D](https://github.com/open-mmlab/mmdetection3d/blob/main/docs/en/model_zoo.md) library.
Please follow the [Installation](https://mmdetection3d.readthedocs.io/en/latest/get_started.html#installation)
and set up your environment.

## Usage

All data for this example integration lives in the publicly accessible S3 bucket `s3://kolena-public-examples`.

First, ensure that the `KOLENA_TOKEN` environment variable is populated in your environment. See our
[initialization documentation](https://docs.kolena.com/installing-kolena/#initialization) for details.

This project defines two scripts that perform the following operations:

1. [`upload_dataset.py`](object_detection_3d/upload_dataset.py) uploads
   the [KITTI](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset.

```shell
$ poetry run python3 object_detection_3d/upload_dataset.py --help
usage: upload_dataset.py [-h] [--dataset DATASET]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  Optionally specify a custom dataset name to upload.
```

2. [`upload_results.py`](object_detection_3d/upload_results.py) uploads results for one of the following
   models: `parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-3class`, and `pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class`.

The `upload_results.py` script defines command line arguments to select which model to evaluate — run using the
`--help` flag for more information:

```shell
$ poetry run python3 object_detection_3d/upload_results.py --help
usage: upload_results.py [-h] [--dataset DATASET]
                         {parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-3class,pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class}

positional arguments:
  {parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-3class,pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class}
                        Name of the model to test.

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Optionally specify a custom dataset name to test.
```

## Quality Standards Guide

Once the dataset and results have been uploaded to Kolena, visit [Kolena](https://app.kolena.com/redirect/) to
[explore the data and results](https://docs.kolena.com/dataset/quickstart/#step-3-explore-data-and-results).

Here are our [Quality Standards](https://docs.kolena.com/dataset/core-concepts/quality-standard/) recommendations for 3D
Object Detection:

### Metrics

1. [Precision](https://docs.kolena.com/metrics/precision)
2. [Recall](https://docs.kolena.com/metrics/recall)
3. [F1-score](https://docs.kolena.com/metrics/f1-score)

### Plots

1. Distribution: `datapoint.image_bboxes[].label`
2. Distribution: `result.TP_3D[].label`
3. Distribution: `result.FP_3D[].label`
4. Distribution: `result.FN_3D[].label`
5. Distribution: `result.nMismatchedInferences`
6. `datapoint.total_objects` vs. `mean(nMissedObjects)`
7. `datapoint.total_objects` vs. `mean(nMismatchedInferences)`
